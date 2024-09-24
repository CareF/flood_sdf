import numpy as np
import numpy.linalg
from numpy.typing import ArrayLike, NDArray
import osqp  # QP solver
from scipy import sparse
import trimesh


def point_to_triangle_linarg(point: ArrayLike, triangle: NDArray):
    """Return the distance from a point to a triangle.

    Experimental, not part of the work
    """
    # x * (r1, r2, r3, norm) == point & sum(x[:3]) == 1
    norm = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[1])
    norm = norm / numpy.linalg.norm(norm)
    x = np.linalg.solve(
        np.block([[triangle.T, norm.reshape((3, 1))], [np.ones((1, 3)), 0]]),
        np.append(point, 1),
    )
    a, b, c, h = x
    print(a, b, c, h)
    if a >= 0 and b >= 0 and c >= 0:
        return h
    if a < 0 and b < 0:
        return np.linalg.norm(point - triangle[2])
    if a < 0 and c < 0:
        return np.linalg.norm(point - triangle[1])
    if b < 0 and c < 0:
        return np.linalg.norm(point - triangle[0])
    if a < 0:
        pass
    if b < 0:
        pass
    if c < 0:
        pass
    raise NotImplementedError("TODO")


def point_to_triangle(point: ArrayLike, triangle: NDArray):
    """triangle[i] is the i-th vertex of the triangle, c.c.w. to point out.

    Experimental, not part of the work
    """
    P = sparse.csc_matrix(triangle @ triangle.T)
    q = -triangle @ point
    A = sparse.csc_matrix(np.vstack([np.ones((1, 3)), np.identity(3)]))
    lower = np.array([1.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, np.inf, np.inf, np.inf])
    prob = osqp.OSQP()
    prob.setup(P, q, A, lower, upper, verbose=False)
    res = prob.solve()
    assert res.info.status == "solved"
    residual_vec = point - res.x @ triangle
    return np.linalg.norm(residual_vec)


def close_triangle_distance_sq(
    triangle: NDArray,
    r0: NDArray,
    step: NDArray,
    size: tuple,
    width: int,
    df_sq: NDArray,
):
    """
    trangle[j] is the j-th vertex (ccw to point out) of the triangle.
    grid is r0 + (i, j, k) @ step, where r0 and step are 3D vectors.
    width is the width of how close  to the triangle.

    update df_sq (distance field squared) in place.
    """
    norm = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[1])
    norm = norm / numpy.linalg.norm(norm)
    start = ((triangle[0] - r0) / step).astype(int)
    if np.any(start < 0) or np.any(start >= size):
        raise ValueError("Triangle out of grid.")
    start = tuple(start)

    # QP problem: min 1/2 x^T P x + q^T x, s.t. l <= Ax <= u
    p_mat = sparse.csc_matrix(triangle @ triangle.T)
    q_vec = -triangle @ triangle[0]
    a_mat = sparse.csc_matrix(np.vstack([np.ones((1, 3)), np.identity(3)]))
    lower = np.array([1.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, np.inf, np.inf, np.inf])
    prob = osqp.OSQP()
    prob.setup(p_mat, q_vec, a_mat, lower, upper, verbose=False)

    processed = set()
    to_be_calc = [start]
    while to_be_calc:
        point_idx = to_be_calc.pop()
        processed.add(point_idx)
        point = r0 + step * point_idx
        q_vec = -triangle @ point
        prob.update(q=q_vec)
        res = prob.solve()
        assert res.info.status == "solved"
        residual_vec = point - res.x @ triangle
        if np.any(np.abs(residual_vec) > width * step):
            continue
        # distance_sq = 2 * res.info.obj_val + point @ point
        distance_sq = residual_vec @ residual_vec
        # +- is still problematic near a acute angle. Maybe ignore this &
        # use a global +- sign afterward.
        # distance_to_plane = residual_vec @ norm
        # if distance_to_plane < 0:
        #     distance = -np.linalg.norm(residual_vec)
        # else:
        #     distance = np.linalg.norm(residual_vec)
        # distance = np.linalg.norm(residual_vec)
        if df_sq[point_idx] > distance_sq:
            df_sq[point_idx] = distance_sq

        for idx in point_idx + np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ):
            if np.any(idx < 0) or np.any(idx >= size):
                continue
            idx = tuple(idx)
            if idx in processed:
                continue
            to_be_calc.append(idx)


def get_sign(
    mesh: trimesh.Trimesh, r0: NDArray, step: NDArray, size: tuple, width: int
):
    xs, ys, zs = [
        np.arange(r0[i], r0[i] + step[i] * size[i], step[i]) for i in range(3)
    ]
    grids = np.moveaxis(np.array(np.meshgrid(xs, ys, zs, indexing="ij")), 0, -1)
    return 1 - mesh.contains(grids.reshape((-1, 3))).reshape(grids.shape[:-1]) * 2


def close_mesh_sdf(
    mesh: trimesh.Trimesh, r0: NDArray, step: NDArray, size: tuple, width: int
):
    """
    trangles[i][j] is the j-th vertex (ccw to point out) of the i-th triangle.
    grid is r0 + (i, j, k) @ step, where r0 and step are 3D vectors.
    """
    assert width >= 1
    sdf = np.inf * np.ones(size)
    for triangle in mesh.triangles:
        close_triangle_distance_sq(triangle, r0, step, size, width, sdf)
    return np.sqrt(sdf) * get_sign(mesh, r0, step, size, width)


if __name__ == "__main__":
    point = np.array([2.0, 0.0, 0.0])
    triangle = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])
    print(point_to_triangle(point, triangle))
    print(point_to_triangle_linarg(point, triangle))
