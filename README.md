# three-pointcloud-raycast

A drop-in replacement for `THREE.Raycaster` that raycasts **Potree point clouds** much faster, while returning **exactly** the same hits: the same points, the same values bit for bit, in the same order.

The stock raycaster tests every point of every octree node the ray's bounding sphere touches. On a cloud with 1.5M points, one ray can test hundreds of thousands of points. This library uses a two-level BVH, the idea behind [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh), so it tests only the few points near the ray.

## Install

```bash
npm install three-pointcloud-raycast three
```

`three` is a peer dependency, so the library always uses your app's copy of three.js.

## Usage

```ts
import { RaycastPointCloud } from 'three-pointcloud-raycast';

const raycaster = new RaycastPointCloud();
raycaster.params.Points = { threshold: 0.4 };
raycaster.set(origin, direction); // direction normalized
const hits = raycaster.intersectObject(pointCloud);       // or
const all = raycaster.intersectObjects(scene.children, true);
```

`RaycastPointCloud` extends `THREE.Raycaster`, so `set`, `setFromCamera`, `params`, `layers`, `near`, `far` and `camera` all work as usual. You can swap it in anywhere you use a `THREE.Raycaster`.

### Points inside a box

```ts
// Points of the clouds inside `box`, where `box` is in `slab`'s local space,
// returned in slab-local coordinates.
const points = RaycastPointCloud.getPointsInLocalBox(clouds, slab, box);
```

This gives the same result as converting every point to world space, calling `slab.worldToLocal`, and filtering with `box.containsPoint`. The difference is that it skips octree nodes that can't reach the box, and only allocates the points it keeps.

### Lower-level API

| Export | What it does |
|---|---|
| `intersectPointCloud(raycaster, object, recursive?, target?)` | Same as `raycaster.intersectObject`, accelerated. Works with any `THREE.Raycaster`. |
| `intersectPointClouds(raycaster, objects, recursive?, target?)` | Same as `raycaster.intersectObjects`, accelerated. |
| `getPointCloudPointsInLocalBox(clouds, localObject, box)` | The box query above. |
| `PointsBVH`, `computeFinitePointBounds` | The point BVH, for your own queries. |

## What is accelerated

- **Accelerated:** the octree nodes of a Potree `PointCloudOctree`. Clouds are recognised by their shape (`pcoGeometry` plus `root.sceneNode`), so this library doesn't depend on Potree. Once a Potree cloud has been seen, other `Points` objects of the same class are accelerated too.
- **Raycast exactly as three.js does it:** everything else. That includes meshes, lines, sprites, indexed or interleaved point geometry, a non-default `drawRange`, projective matrices, `recursive = false`, and objects with a custom `raycast` method. You can mix all of these in one call.

## Caching

- **Per-geometry point BVH:** built the first time a ray reaches the node, and kept in a `WeakMap`, so it's freed together with the geometry. It is rebuilt when the position attribute's `version` changes (`attribute.needsUpdate = true`).
- **Per-cloud node index:** reused only within one synchronous run, and dropped on the next microtask. Potree only adds, removes or moves nodes in its own callbacks, so the index can't go stale. It is also rebuilt if the cloud's root matrix changes.

## Supported versions

Tested with three.js r161. The peer range is `>=0.161.0`. CI also runs the test suite against the latest three.js to catch breaking changes.

## Development

```bash
npm install
npm test            # equivalence tests against THREE.Raycaster
npm run typecheck
npm run build       # ESM + CJS + .d.ts into dist/
npm run check:pkg   # publint + are-the-types-wrong
```

The main test, `test/raycast.equivalence.test.ts`, fires thousands of random rays at synthetic Potree-shaped clouds. It asserts that the result is identical to `THREE.Raycaster`'s, including transforms, near/far, layers and mixed scenes. Any change to the algorithms must keep it passing.

## License

MIT
