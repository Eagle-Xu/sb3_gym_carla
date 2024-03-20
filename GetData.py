import random
import time
from PIL import Image
import numpy as np
import carla
import pygame
from gym import spaces
from skimage.transform import resize
from agents.carla_agents.navigation.behavior_agent import BehaviorAgent
from environments.gym_carla.misc import rgb_to_display_surface, display_to_rgb
from environments.gym_carla.render import BirdeyeRender
from environments.gym_carla.route_planner import RoutePlanner


class LookWaypoint():

    def __init__(self):

        self.begin=[[0.0,0.0,0.0]]
        self.dest=carla.Location(x=-7.694724, y=29.346333, z=0.000000)
        self.ego=None
        self.obs_range = 32
        self.lidar_bin = 0.125
        self.d_behind=12
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.display_size=256
        self.eposides_max=300
        self.eposide=0
        self.eposide_times=0
        self.sensor_list=[]
        self.number_of_vehicles=100
        self.max_past_step=1
        self.display_route=True
        self.max_waypt=12
        self.max_time_episode=400
        self.dt=0.05

        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'observation': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }
        self.observation_space = spaces.Dict(observation_space_dict)

        print('connecting to Carla server...')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
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

        # Camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=1.7, z=2.4))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '5000')


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
            (self.display_size*2, self.display_size),
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
        self.lidar_sensor = None
        self.camera_sensor = None
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

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))
        self.sensor_list.append(self.lidar_sensor)
        def get_lidar_data(data):
            # data.save_to_disk(os.path.join('out/lidar', '%06d.ply' % data.frame))
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda image: get_camera_img(image))
        self.sensor_list.append(self.camera_sensor)
        def get_camera_img(data):
            # data.save_to_disk(os.path.join('out/camera', '%06d.png' % data.frame))
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Update timesteps
        self.eposide += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        self.begin_time = 0
        self.begin_time = time.time()
        return self._get_obs()

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

        ## Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size, 0))

        # Display on pygame
        pygame.display.flip()

        return birdeye_2,self.camera_img

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
        transform.location.z+=2
        while True:
            if self._try_spawn_ego_vehicle_at(transform):
                self.ego.set_autopilot(True)
                print("第%d次eposide的ego_vehicle 生成成功" % self.eposide)
                break
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


lookwayponit=LookWaypoint()

for i in range(lookwayponit.eposides_max):
     obs ,_= lookwayponit.reset()
     agent = BehaviorAgent(lookwayponit.ego, behavior='normal')
     agent.set_destination(lookwayponit.dest, lookwayponit.begin)
     lookwayponit.eposide_times = 0
     # lookwayponit.draw_waypoints(lookwayponit.waypoints_max, road_id=16, life_time=20)
     while True:

        # 随时更新agent的相关信息，主要是速度相关
        agent._update_information()
        lookwayponit.world.tick()

        #更新pygame
        obs,camer=lookwayponit._get_obs()

        img=Image.fromarray(obs,'RGB')
        img=img.transpose(Image.FLIP_LEFT_RIGHT)#水平翻转
        img=img.rotate(90)#顺时针旋转90度
        img.save("out/dataset/{}.{}___.jpg".format(i+1,lookwayponit.eposide_times))
        #获取位置
        transform=lookwayponit.ego.get_transform()
        if np.sqrt((transform.location.x - lookwayponit.dest.x) ** 2 + (transform.location.y - lookwayponit.dest.y) ** 2) < 4:
            print('===================Success, Arrivied at Target Point!')
            break
        #达到最大步长
        if lookwayponit.eposide_times > lookwayponit.max_time_episode:
            break
        #发生碰撞
        if len(lookwayponit.collision_hist) > 0:
            break
        lookwayponit.eposide_times = lookwayponit.eposide_times+1
        # 设置观察者的视角,设置为正上方观看,可以调用cv2的包生成一个新的窗口来观看
        spectator = lookwayponit.world.get_spectator()
        transform = lookwayponit.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                carla.Rotation(pitch=-90)))

        speed_limit = lookwayponit.ego.get_speed_limit()
        agent.get_local_planner().set_speed(speed_limit)

        control = agent.run_step(debug=True)
        lookwayponit.ego.apply_control(control)






