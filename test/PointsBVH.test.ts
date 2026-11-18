import { describe, expect, it } from 'vitest';

import { computeFinitePointBounds, PointsBVH } from '../src/PointsBVH';
import { mulberry32 } from './helpers';

// The same arithmetic as three.js Ray.distanceSqToPoint.
function bruteForce(
    positions: ArrayLike<number>,
    count: number,
    o: number[],
    d: number[],
    thresholdSq: number,
): number[] {
    const [ox, oy, oz] = o;
    const [dx, dy, dz] = d;
    const out: number[] = [];
    for (let i = 0; i < count; i++) {
        const px = positions[i * 3];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        const dd = (px - ox) * dx + (py - oy) * dy + (pz - oz) * dz;
        let distSq: number;
        if (dd < 0) {
            const ex = ox - px;
            const ey = oy - py;
            const ez = oz - pz;
            distSq = ex * ex + ey * ey + ez * ez;
        } else {
            const fx = dx * dd + ox - px;
            const fy = dy * dd + oy - py;
            const fz = dz * dd + oz - pz;
            distSq = fx * fx + fy * fy + fz * fz;
        }
        if (distSq < thresholdSq) out.push(i);
    }
    return out;
}

function randomDirection(rng: () => number): number[] {
    const d = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
    const l = Math.hypot(d[0], d[1], d[2]);
    return d.map((v) => v / l);
}

describe('computeFinitePointBounds', () => {
    it('ignores NaN and Infinity points', () => {
        const p = [1, 2, 3, NaN, 0, 0, -1, 5, 0, Infinity, 1, 1];
        expect(computeFinitePointBounds(p, 4)).toEqual([-1, 2, 0, 1, 5, 3]);
    });

    it('returns null when no point is finite', () => {
        expect(computeFinitePointBounds([NaN, 0, 0], 1)).toBeNull();
        expect(computeFinitePointBounds([], 0)).toBeNull();
    });
});

describe('PointsBVH.collectRayHits', () => {
    it.each([
        ['default leaves', {}],
        ['tiny leaves (deep tree)', { maxLeafSize: 2 }],
    ])('matches brute force with %s', (_, options) => {
        const rng = mulberry32(31);
        const count = 5000;
        const positions = new Float32Array(count * 3);
        for (let i = 0; i < positions.length; i++) {
            positions[i] = (rng() - 0.5) * 50;
        }
        positions[301] = NaN;
        const bvh = new PointsBVH(positions, count, options);
        const out = { array: new Uint32Array(1) };

        for (let r = 0; r < 200; r++) {
            const o = [(rng() - 0.5) * 80, (rng() - 0.5) * 80, (rng() - 0.5) * 80];
            const d = randomDirection(rng);
            const threshold = 0.2 + rng() * 2;
            const n = bvh.collectRayHits(
                o[0],
                o[1],
                o[2],
                d[0],
                d[1],
                d[2],
                threshold * (1 + 1e-6),
                Infinity,
                threshold * threshold,
                out,
            );
            const got = Array.from(out.array.subarray(0, n)).sort(
                (a, b) => a - b,
            );
            expect(got).toEqual(
                bruteForce(positions, count, o, d, threshold * threshold),
            );
        }
    });

    it('handles axis-parallel rays', () => {
        const positions = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 5, 5]);
        const bvh = new PointsBVH(positions, 4, { maxLeafSize: 1 });
        const out = { array: new Uint32Array(4) };
        const n = bvh.collectRayHits(-10, 0, 0, 1, 0, 0, 0.5, Infinity, 0.25, out);
        expect(Array.from(out.array.subarray(0, n)).sort()).toEqual([0, 1]);
    });

    it('handles empty and all-NaN inputs', () => {
        const out = { array: new Uint32Array(4) };
        expect(
            new PointsBVH(new Float32Array(0), 0).collectRayHits(
                0, 0, 0, 1, 0, 0, 1, Infinity, 1, out,
            ),
        ).toBe(0);
        expect(
            new PointsBVH(new Float32Array([NaN, NaN, NaN]), 1).collectRayHits(
                0, 0, 0, 1, 0, 0, 1, Infinity, 1, out,
            ),
        ).toBe(0);
    });

    it('handles a single point and coincident points', () => {
        const out = { array: new Uint32Array(1) };
        const single = new PointsBVH(new Float32Array([2, 0, 0]), 1);
        expect(single.collectRayHits(0, 0, 0, 1, 0, 0, 1, Infinity, 1, out)).toBe(1);

        const same = new Float32Array(300).fill(3);
        const bvh = new PointsBVH(same, 100, { maxLeafSize: 4 });
        const n = bvh.collectRayHits(0, 0, 0, 1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3), 1, Infinity, 1, out);
        expect(n).toBe(100);
    });
});
