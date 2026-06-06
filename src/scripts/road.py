"""
CARLA - 列出所有 road_id 对应的 spawn_point_index 和坐标
适用任意地图
"""

import carla
from collections import defaultdict


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()  # 替换为你使用的地图名称
    world = client.load_world('Town03')
    carla_map = world.get_map()
    #切换Town03
    


    spawn_transforms = carla_map.get_spawn_points()
    road_to_spawns = defaultdict(list)

    for i, tr in enumerate(spawn_transforms):
        wp = carla_map.get_waypoint(tr.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp:
            road_to_spawns[wp.road_id].append({
                'spawn_index': i,
                'x': round(tr.location.x, 3),
                'y': round(tr.location.y, 3),
                'z': round(tr.location.z, 3),
            })

    print(f"{'road_id':>8}  {'spawn_index':>12}  {'x':>10}  {'y':>10}  {'z':>8}")
    print("-" * 60)
    for road_id in sorted(road_to_spawns.keys()):
        for sp in road_to_spawns[road_id]:
            print(f"{road_id:>8}  {sp['spawn_index']:>12}  {sp['x']:>10}  {sp['y']:>10}  {sp['z']:>8}")


if __name__ == '__main__':
    main()