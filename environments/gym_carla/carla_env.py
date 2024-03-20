import random
import time
import gymnasium as gym

import torch
from PIL import Image
import numpy as np
import carla
import pygame
from gymnasium import spaces
import copy
from skimage.transform import resize
from agents.carla_agents.navigation.behavior_agent import BehaviorAgent
from environments.gym_carla.misc import rgb_to_display_surface, display_to_rgb
from environments.gym_carla.render import BirdeyeRender
from environments.gym_carla.route_planner import RoutePlanner
from environments.gym_carla.misc import *
import torchvision
from environments.gym_carla.models import Encoder

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")


class CarlaEnv(gym.Env):

    def __init__(self):

        self.begin=[[0.0,0.0,0.0]]

        self.destsEntransLeft=carla.Location(x=24.199064, y=-5.089221, z=-0.001813) #到达圆盘入口左
        self.destsEntransRight = carla.Location(x=24.205204, y=-8.726452, z=-0.000727)  # 到达圆盘入口右
        self.counts_entrans = 0.0  # 到达次数
        self.control = 0.0 # 到一次记一次

        self.destsOneZero=carla.Location(x=6.867558, y=-23.746708, z=0.000000) #第一个出口右路
        self.destsOneOne=carla.Location(x=10.130323, y=-25.013346, z=0.000000)#第一个出口左路
        self.destsOneZero1 = carla.Location(x=8.339429, y=-17.410095, z=0.000000)  # 第一个出口近云盘右路
        self.destsOneOne1 = carla.Location(x=9.950678, y=-20.517166, z=0.000000)  # 第一个出口近云盘左路
        self.counts_first = 0.0  # 到达第一个路口的次数
        self.control1 = 0.0

        self.destsTowFirst=carla.Location(x=-24.282610, y=-9.487700, z=0.000000) # 第二个出口前一个waypoint
        self.destsTowSecond = carla.Location(x=-25.053129, y=-8.880978, z=0.000000)# 第二个出口后一个waypoint
        self.destsTowFirst1 = carla.Location(x=-23.871666, y=-1.452438, z=0.000000)  # 第二个出口近云盘右路
        self.destsTowSecond1 = carla.Location(x=-20.383087, y=-1.169919, z=0.000000)  # 第二个出口近云盘左路
        self.counts_second=0.0 #到达第二个路口次数
        self.control2=0.0

        self.destsThreeFirst = carla.Location(x=-2.181200, y=23.763405, z=0.000000)  # 第三个出口近云盘右路
        self.destsThreeSecond = carla.Location(x=-1.921998, y=20.273014, z=0.000000)  # 第三个出口近云盘左路
        self.destsThreeFirst1 = carla.Location(x=-7.523384, y=30.356438, z=0.000000)  # 第三个出口左
        self.destsThreeSecond1 = carla.Location(x=-10.978111, y=30.917557, z=0.000000)  # 第三个出口右
        self.counts_three = 0.0  # 到达第三个路口次数
        self.control3 = 0.0

        self.dests = carla.Location(x=-10.067160, y=43.628490, z=0.000000)  # 目标点
        self.dests1 = carla.Location(x=-6.567230, y=43.606346, z=0.000000)
        self.counts_target = 0.0  # 到达目标点的次数

        self.ego=None
        self.obs_range = 32
        self.lidar_bin = 0.125
        self.d_behind=12
        self.obs_size =64
        self.display_size=256
        self.eposides_max=300 #最大轮数
        self.eposide=0 #记录第几轮
        self.eposide_times=0 #记录每一轮的步数
        self.max_time_episode = 400  #每一轮的最大步数
        self.sensor_list=[]
        self.number_of_vehicles=100
        self.max_past_step=1
        self.display_route=True
        self.max_waypt=12
        self.max_time_episode=900
        self.dt=0.05
        self.out_lane_thres = 2
        self.prev_action = [0.0, 0.0]
        self.action_lambda = 0.5
        self.model = torch.load("The_300_eposide_encoder_model.pth")
        self.max_ego_spawn_times=200
        self.k=0.0#跳帧行动
        self.control=0.0

        self.action_space = spaces.box.Box(np.array([0.0,-1.0]),
                                       np.array([1.0,1.0]), dtype=np.float32) # acc,ster,b
        self.observation_space = spaces.box.Box(low=0, high=255,shape=(1,64),dtype=np.float32)

        print('connecting to Carla server...')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world("Town03")
        print('Carla server connected!')

        # 以距离为1的间距创建waypoints
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.waypoints = self.world.get_map().generate_waypoints(distance=1.0)

        #获取蓝图
        self.blueprint_library = self.world.get_blueprint_library().filter("vehicle.lincoln*")
        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint('vehicle.lincoln*', color='255,0,0')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')


        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt #秒

        # Initialize the renderer
        self._init_renderer()

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
        """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def reset(self):
        # Clear sensor objects  重置传感器
        self.collision_sensor = None
        self.waypoints = self.world.get_map().generate_waypoints(distance=1.0)

        # Delete sensors, vehicles and walkers # 消除所有对象
        self._clear_all_actors(['vehicle.*', 'controller.ai.walker', 'walker.*'])
        if len(self.sensor_list) != 0:
            for sensor in self.sensor_list:
                sensor.destroy()
        self.sensor_list = []

        # Disable sync mode  取消同步模式
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)  # 打乱出生点
        count = self.number_of_vehicles  # 获取车辆的数量
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)

        # create the ego vehicle
        self.set_vehicle_begain(self.waypoints,8)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        self.sensor_list.append(self.collision_sensor)
        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []

        # Update timesteps
        self.eposide += 1
        self.eposide_times = 0
        self.prev_action = [0.0, 0.0]
        self.control = 0.0
        self.control1 = 0.0
        self.control2 = 0.0
        self.control3 = 0.0

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        self.begin_time = 0
        self.begin_time = time.time()

        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front
        }

        return (self._get_obs(),copy.deepcopy(info))

    def step(self, action):
        # Calculate acceleration and steering  # 计算加速度和转向
        # if self.k == 0.0:
        #    self.prev_action[0] = action[0]
        #    self.prev_action[1] = action[1]
        #    acc = action[0]
        #    steer = action[1]
        #    self.k=self.k+1
        #
        # elif self.k == 4:
        #     acc = action[0]
        #     steer = action[1]
        #     self.k = 0.0
        # else:
        #     acc = self.prev_action[0]
        #     steer = self.prev_action[1]
        #     self.k = self.k + 1
        acc = action[0]
        steer = action[1]
        # brake = action[2]

        act = carla.VehicleControl(steer=float(steer), throttle=float(acc), brake=0.0)  # act = carla.VehicleControl(steer=0.0, throttle=1.0 ,brake=0.0)
        self.ego.apply_control(act)

        self.world.tick()  # 更新simulation

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front
        }

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:  # 大于最大
            self.vehicle_polygons.pop(0)
            # state information

        self.eposide_times= self.eposide_times+1

        return (self._get_obs(), self._get_reward(), self._terminal(),False, copy.deepcopy(info))

    def _get_reward(self):
        """Calculate the step reward."""
        # speed
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        if speed > 5:
            speed = 10 - speed

        # the magnitude of steering
        r_steer = 0.5 * self.ego.get_control().steer ** 2

        # collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -10

        # running out of the lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1
        r_1=0.0
        if np.sqrt((ego_x - self.destsOneZero.x) ** 2 + (ego_y - self.destsOneZero.y) ** 2) < 2 or np.sqrt(
                (ego_x - self.destsOneOne.x) ** 2 + (ego_y - self.destsOneOne.y) ** 2) < 2:
            r_1 = -50
        r_2=0.0
        if np.sqrt((ego_x - self.destsTowFirst.x) ** 2 + (ego_y - self.destsTowFirst.y) ** 2) < 2 or np.sqrt(
                (ego_x - self.destsTowSecond.x) ** 2 + (ego_y - self.destsTowSecond.y) ** 2) < 2:
            r_2=-50
        # 到达目的地
        r_dests=0
        if np.sqrt((ego_x - self.dests.x) ** 2 + (ego_y - self.dests.y) ** 2) < 4:
            r_dests = 50

        return speed + r_steer + r_collision + r_out + r_dests +r_1 +r_2 - 0.1

    def _get_reward1(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        r_speed = -abs(speed - 8.0)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > 8.0:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon ** 2

        r = r_speed+200 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1

        return r
    def _get_obs(self):

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:  # 大于最大
            self.vehicle_polygons.pop(0)

        # route planner
        if self.eposide_times %10 == 0:
            self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
            self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
        """Get the observations"""
        # Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons  # 100个（4，2）数组，全部状态
        self.birdeye_render.waypoints = self.waypoints  # 12个点

        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:  # 是否显示道路
            birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye_1 = pygame.surfarray.array3d(self.display)
        birdeye_2 = birdeye_1[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye_2, self.obs_size)

        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        self.display.blit(birdeye_surface, (0, 0))



        # Display on pygame
        pygame.display.flip()

        img = Image.fromarray(birdeye_2, 'RGB')
        img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
        img = img.rotate(90)
        img = torchvision.transforms.Resize([255, 255])(img)
        img = torchvision.transforms.ToTensor()(img)

        img = img.to(device)
        img = self.model(img)

        img = img[2].squeeze(0)
        img = img.detach().cpu().numpy()

        # v = self.ego.get_velocity()
        # speed = np.float32(np.sqrt(v.x ** 2 + v.y ** 2))#速度
        # speed=np.array(speed)
        # img=np.append(img,speed)
        return img

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            # print("碰撞位置:")
            # print(self.ego.get_transform())
            return True

        # If reach maximum timestep
        if self.eposide_times > self.max_time_episode:
            return True

        #到达圆盘入口
        if self.control==0.0:
            if np.sqrt((ego_x - self.destsEntransLeft.x) ** 2 + (ego_y - self.destsEntransLeft.y) ** 2) < 2  or np.sqrt((ego_x - self.destsEntransRight.x) ** 2 + (ego_y - self.destsEntransRight.y) ** 2) < 2:
                self.counts_entrans = self.counts_entrans + 1
                print("到达圆盘入口{}次,占比%{}".format(self.counts_entrans,(self.counts_entrans/self.eposide)*100))
                self.control =1

        #到达第一个出口
        if np.sqrt((ego_x - self.destsOneZero.x) ** 2 + (ego_y - self.destsOneZero.y) ** 2) < 2 or np.sqrt( (ego_x - self.destsOneOne.x) ** 2 + (ego_y - self.destsOneOne.y) ** 2) < 2:  # 到达第一个出口
            if self.control1 == 0:
               self.counts_first = self.counts_first+1
               print("到达圆盘第一个出口{}次,占比%{}".format(self.counts_first,(self.counts_first/self.eposide)*100))
               self.control1 = 1
            return True
        if self.control1 == 0:
            if np.sqrt((ego_x - self.destsOneZero1.x) ** 2 + (ego_y - self.destsOneZero1.y) ** 2) < 4 or np.sqrt(
                    (ego_x - self.destsOneOne1.x) ** 2 + (ego_y - self.destsOneOne1.y) ** 2) < 4:  # 到达第一个出口
                self.counts_first = self.counts_first + 1
                self.control1 = 1
                print("到达圆盘第一个出口{}次,占比%{} ".format(self.counts_first,(self.counts_first/self.eposide)*100))


        #到达第二个出口
        if np.sqrt((ego_x - self.destsTowFirst.x) ** 2 + (ego_y - self.destsTowFirst.y) ** 2) < 2 or np.sqrt(
                (ego_x - self.destsTowSecond.x) ** 2 + (ego_y - self.destsTowSecond.y) ** 2) < 2:
            if self.control2 == 0:
               self.counts_second = self.counts_second+1
               self.control2=1
               print("到达圆盘第二个出口{}次,占比%{}".format(self.counts_second,(self.counts_second/self.eposide)*100))
            return True
        if self.control2 == 0:
            if np.sqrt((ego_x - self.destsTowFirst1.x) ** 2 + (ego_y - self.destsTowFirst1.y) ** 2) < 4 or np.sqrt(
                    (ego_x - self.destsTowSecond1.x) ** 2 + (ego_y - self.destsTowSecond1.y) ** 2) < 4:
                self.counts_second = self.counts_second + 1
                self.control2 = 1
                print("到达圆盘第二个出口{}次,占比%{}".format(self.counts_second,(self.counts_second/self.eposide)*100))

        #到达第3个出口
        if np.sqrt((ego_x - self.destsThreeFirst.x) ** 2 + (ego_y - self.destsThreeFirst.y) ** 2) < 2 or np.sqrt(
                (ego_x - self.destsThreeSecond.x) ** 2 + (ego_y - self.destsThreeSecond.y) ** 2) < 2:
            self.counts_three= self.counts_three + 1
            print("到达圆盘三个出口{}次,占比%{}".format(self.counts_three,(self.counts_three/self.eposide)*100))
            return True
        if self.control3 == 0:
            if np.sqrt((ego_x - self.destsThreeFirst1.x) ** 2 + (ego_y - self.destsThreeFirst1.y) ** 2) < 4 or np.sqrt(
                    (ego_x - self.destsThreeSecond1.x) ** 2 + (ego_y - self.destsThreeSecond1.y) ** 2) < 4:
                self.counts_three = self.counts_three + 1
                self.control3 = 1
                print("到达圆盘三个出口{}次,占比%{}".format(self.counts_three,(self.counts_three/self.eposide)*100))

        # If at destination
        if np.sqrt((ego_x - self.dests.x) ** 2 + (ego_y - self.dests.y) ** 2) < 4 or np.sqrt((ego_x - self.dests1.x) ** 2 + (ego_y - self.dests1.y) ** 2) < 4:
            self.counts_target = self.counts_target + 1
            print("到达目标{}次,占比%{}".format(self.counts_target,(self.counts_target/self.eposide)*100))
            return True


        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            # print("出线位置:")
            # print(self.ego.get_transform())
            return True

        return False

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous(同步) mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(True)
            return True
        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw（侧滑角） of the actor
            trans = actor.get_transform()  # 获取坐标
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate（局部坐标）
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate（全局坐标）
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':  # aiwalker
                        actor.stop()
                    actor.destroy()
                    self.world.get_actors()

    def draw_waypoints(self,waypoints, road_id=None, life_time=500.0):
        for waypoint in waypoints:
            if (waypoint.road_id == road_id):
                self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                        persistent_lines=True)

    def set_vehicle_begain(self,waypoints,road_id):
        filtered_waypoints = []
        for waypoint in waypoints:
            if (waypoint.road_id == road_id):
                 filtered_waypoints.append(waypoint)
        # print(len(filtered_waypoints))
        transform=filtered_waypoints[41].transform
        transform.location.z=+1
        ego_spawn_times=0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                print("达到生成限制时间")
                ego_spawn_times = 0
                self._clear_all_actors(['vehicle.*', 'controller.ai.walker', 'walker.*'])
                if len(self.sensor_list) != 0:
                    for sensor in self.sensor_list:
                        sensor.destroy()
                self.sensor_list = []
            if self._try_spawn_ego_vehicle_at(transform):
                # self.ego.set_autopilot(True)
                # print("第%d次eposide的ego_vehicle 生成成功" % self.eposide)
                break
            ego_spawn_times += 1
            time.sleep(0.1)
            vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
            self.vehicle_polygons.append(vehicle_poly_dict)
            while len(self.vehicle_polygons) > self.max_past_step:  # 大于最大
                self.vehicle_polygons.pop(0)
        self.begin = transform

        # 设置观察者的视角,设置为正上方观看,可以调用cv2的包生成一个新的窗口来观看
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                carla.Rotation(pitch=-90)))







