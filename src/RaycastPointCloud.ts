import * as THREE from 'three';

import { getPointCloudPointsInLocalBox } from './pointCloudBoxQuery';
import { intersectPointCloud, intersectPointClouds } from './pointCloudRaycast';

/**
 * A THREE.Raycaster that raycasts Potree point clouds fast.
 *
 * Use it exactly like a THREE.Raycaster:
 *
 *   const raycaster = new RaycastPointCloud();
 *   raycaster.params.Points = { threshold: 0.4 };
 *   raycaster.set(origin, direction);
 *   const hits = raycaster.intersectObjects(scene.children, true);
 *
 * `intersectObject` and `intersectObjects` return exactly what the stock
 * THREE.Raycaster returns: the same hits, with the same values, in the same
 * order. Only the way they are found differs:
 *  - Potree octree nodes are searched with a two level BVH (octree node boxes,
 *    then a PointsBVH per node geometry), so only the few points near the ray
 *    are tested instead of every point of every node the ray touches.
 *  - Every other object (meshes, lines, plain THREE.Points, ...) is raycast
 *    with its own `raycast` method, exactly like three.js does, so mixing
 *    point clouds and other objects in one call is fine.
 *
 * Everything else (set, setFromCamera, params, layers, near, far, camera) is
 * the inherited THREE.Raycaster behaviour, so an instance can replace a
 * THREE.Raycaster anywhere.
 */
export class RaycastPointCloud extends THREE.Raycaster {
    /**
     * Same as THREE.Raycaster#intersectObject, BVH accelerated for point
     * clouds.
     *
     * @param object point cloud (or any object)
     * @param recursive also test the descendants
     * @param intersects array the hits are appended to
     * @returns intersections sorted by distance
     */
    override intersectObject<
        TIntersected extends THREE.Object3D = THREE.Object3D,
    >(
        object: THREE.Object3D,
        recursive = true,
        intersects: THREE.Intersection<TIntersected>[] = [],
    ): THREE.Intersection<TIntersected>[] {
        return intersectPointCloud(this, object, recursive, intersects);
    }

    /**
     * Same as THREE.Raycaster#intersectObjects, BVH accelerated for point
     * clouds.
     *
     * @param objects point clouds (or any objects)
     * @param recursive also test the descendants
     * @param intersects array the hits are appended to
     * @returns intersections sorted by distance
     */
    override intersectObjects<
        TIntersected extends THREE.Object3D = THREE.Object3D,
    >(
        objects: THREE.Object3D[],
        recursive = true,
        intersects: THREE.Intersection<TIntersected>[] = [],
    ): THREE.Intersection<TIntersected>[] {
        return intersectPointClouds(this, objects, recursive, intersects);
    }

    /**
     * Points of `pointClouds` that fall inside `localBox`, a box in the local
     * space of `localObject` (e.g. a thin slab around a plane).
     *
     * Same points, order and values as converting every point of every
     * Points node to world space, then `localObject.worldToLocal`, dropping
     * NaN points and keeping those for which `localBox.containsPoint` is
     * true, without allocating every point of the clouds.
     *
     * @param pointClouds clouds to read (Potree or THREE.Points)
     * @param localObject object whose local space `localBox` is in
     * @param localBox box in localObject space
     * @returns kept points in localObject space
     */
    static getPointsInLocalBox(
        pointClouds: readonly THREE.Object3D[],
        localObject: THREE.Object3D,
        localBox: THREE.Box3,
    ): THREE.Vector3[] {
        return getPointCloudPointsInLocalBox(
            pointClouds,
            localObject,
            localBox,
        );
    }
}
