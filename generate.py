import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from qrsolver import solve

torch.set_printoptions(sci_mode=True, precision=5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# debugging error codes have random strings because I can't be bothered to come up with so many names
# TODO:
draw_threshold = 100
"""
1. make generate_hexagon pixel perfect
2. deal with N lines intersecting at a single point
3. to prevent conductances = inf, merge nearby points
"""


def circuit_solve(size, row, col, data, b, reference_intersections):
    print(f"Sparsity: {(data.size(0) / (size ** 2))}")
    print(f"num_intersections: {size+1}")
    print(f"num_elements: {data.size(0)}")
    print("Solving...")
    dx = qr_solve(size, row, col, data, b)

    dx = torch.cat([dx, torch.zeros(1, dtype=torch.double, device=device)], dim=0)
    resistance = dx[reference_intersections[0]] - dx[reference_intersections[1]]
    return dx, resistance


def qr_solve(size, row, col, data, b, tol=1e-10):
    nnz = data.size(0)
    dcsrRow = torch.empty(size + 1, dtype=torch.int, device=device)
    dx = torch.empty(size, dtype=torch.double, device=device)
    singularity = solve(nnz, size, tol, data, col.int(), row.int(), dcsrRow, b, dx)
    if singularity != -1:
        raise ValueError("Error: 6CBP3, singular matrix")
    return dx


# https://stackoverflow.com/questions/20027936/how-to-efficiently-concatenate-many-arange-calls-in-numpy
# Can still be vectorized
def multirange(lower, upper):
    """
    Concatenates multiple aranges

    :param lower tensor: 1d lower bounds
    :param upper tensor: 1d upper bounds
    :output tensor: 1d tensor
    """
    ranges = torch.cat(
        [
            torch.arange(lower[i], upper[i], dtype=torch.long, device=device)
            for i in range(len(lower))
        ]
    )
    return ranges


def generate_rotation(theta):
    """
    Generates rotation matrix for a given theta

    :param theta tensor: 0d rotation angle counterclockwise
    :output tensor: [2,2] tensor
    """
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def generate_hexagon(max_radius, side_length=1, theta=0, epsilon=0):
    """
    Generates a list of all line segments in a hexagonal grid

    :param max_radius float: size of the circle that encompasses the entire hexagonal grid
    :param side_length float: side length of a unit cell hexagon
    :param theta float: rotation angle
    :param epsilon float: length appended to each segment
    :output tensor: [num_segments, 2 points, xy coordinates]
    """
    # Circle Radius (will be surpassed by a bit)
    # assuming the side length = 1
    assert max_radius > 1
    segments = []

    max_radius = torch.as_tensor(max_radius, dtype=torch.double, device=device)
    side_length = torch.as_tensor(side_length, dtype=torch.double, device=device)
    theta = torch.as_tensor(theta, dtype=torch.double, device=device)
    epsilon = torch.as_tensor(epsilon, dtype=torch.double, device=device)

    y_max = torch.floor(max_radius / (1.5 * side_length))  # (1 + 1*sin(30))
    y_min = torch.floor(-max_radius / (1.5 * side_length))  # (1 + 1*sin(30))

    y_steps = torch.arange(y_min, y_max + 1, dtype=torch.double, device=device)
    has_gaps = y_steps % 2
    gaps = has_gaps * (np.sqrt(3) / 2) * side_length

    xs = torch.sqrt(
        (max_radius ** 2)
        - (
            torch.min(
                torch.abs(y_steps * 1.5 * side_length),
                torch.abs((y_steps * 1.5 + 1) * side_length),
            )
            ** 2
        )
    )

    if torch.any((xs - gaps) < 0):
        raise ValueError("Error: 52YJL")

    x_mins = -torch.floor((xs - gaps) / (np.sqrt(3) * side_length))
    x_maxs = -x_mins + has_gaps + 1  # +1 for arange

    x_steps = multirange(x_mins, x_maxs)
    y_steps = torch.repeat_interleave(y_steps, (x_maxs - x_mins).long())
    gaps = torch.repeat_interleave(gaps, (x_maxs - x_mins).long())

    x_1 = x_2 = x_steps * np.sqrt(3) * side_length - gaps
    y_1 = y_steps * 1.5 * side_length - epsilon
    y_2 = y_1 + 1 * side_length + epsilon

    # [3, 2, 2]
    rotations = torch.stack(
        [
            generate_rotation(angle + theta)
            for angle in torch.linspace(
                0, 2 * 2 / 3 * np.pi, steps=3, dtype=torch.double, device=device,
            )
        ]
    )
    # [segments, points, [x,y]]
    segments = torch.stack([torch.stack([x_1, y_1]), torch.stack([x_2, y_2])]).permute(
        2, 0, 1
    )

    # [3, segments, points, [x,y]]
    segments = torch.einsum("abc,dec->adeb", rotations, segments)
    # [segments, points, [x,y]]
    segments = segments.reshape(-1, *segments.shape[-2:])
    return segments


def new_generate_hexagon(max_radius, side_length):
    pass


def generate_bins(segments, max_radius, max_side_length=1):  # , epsilon=1e-5):
    """
    Divides cartesian plane contained within a square (NOT CIRCLE) into bins (ie. square grid).
    Each bin contains indices of all segments that might intersect within that bin, with given that all segments have length less or equal to max_side_length
    An index of (-1) is a filler, and does not refer to any segments

    :param segments tensor: [num_segments, 2 points, xy coordinates]
    :param max_radius float: HALF the length of the square that (encompasses or intersects) all segments
    # NOTE: max_side_length should be computed internally but whatever
    :param max_side_length float: Maximum length of a segment
    :output: [num_bins, num_segments], indices of the segments contained within each bin
    """
    max_radius = torch.as_tensor(max_radius, dtype=torch.double, device=device)
    max_side_length = torch.as_tensor(
        max_side_length, dtype=torch.double, device=device
    )
    # epsilon = torch.as_tensor(epsilon, dtype=torch.double, device=device)

    bin_length = max_side_length * np.sqrt(1 / 2) / 2
    # bin_length = torch.tensor(1,dtype=torch.double, device=device)
    # bin_length = 3

    # bin_dim = 2 * torch.ceil((max_radius + max_side_length) / bin_length)
    # NOTE NOTE NOTE important
    bin_dim = torch.ceil(2 * (max_radius + max_side_length) / bin_length)
    # -- = +
    bin_identity = torch.floor((segments + max_radius + max_side_length) / bin_length)
    if torch.any(bin_identity < 0):
        raise ValueError(f"Error: 5QCYA, {str(torch.min(bin_identity))}")
    bin_identity = bin_identity[..., 1] * bin_dim + bin_identity[..., 0]
    bin_identity = bin_identity.long()

    bin_identity = bin_identity.view(-1)
    if len(segments.shape) == 3:
        segment_identities = torch.arange(
            segments.size(0), dtype=torch.long, device=device
        ).repeat_interleave(2)
    else:
        segment_identities = torch.arange(
            segments.size(0), dtype=torch.long, device=device
        )

    # bins = grid cell, components = segment index
    bins = binning(int(bin_dim ** 2), bin_identity, segment_identities)

    # return spatial dimension of bins
    bins = bins.view([int(bin_dim), int(bin_dim), -1])

    # aggregate between neighboring bins
    neighbors = []
    for y_shift in [-1, 0, 1]:
        for x_shift in [-1, 0, 1]:
            neighbor = torch.roll(bins, shifts=[y_shift, x_shift], dims=[0, 1])
            if y_shift != 0:
                neighbor[int((y_shift - 1) / 2), :] = -1
            if x_shift != 0:
                neighbor[:, int((x_shift - 1) / 2)] = -1
            neighbors.append(neighbor)
    bins = torch.cat(neighbors, dim=-1)
    # remove redundant dimension
    bins = bins.view(int(bin_dim ** 2), -1)

    bins = clean_bins(bins)
    # 0 in bins refers to 0th segment
    return bins


def binning(num_bins, bin_identity, component_identity):
    """
    Given a sequence of bins and a sequence of components with a one-to-one correspondence (ie. component_identity[0] is within bin_identity[0]), put every component_identity within its respective bin_identity within a matrix
    An index of (-1) is a filler, and does not refer to any segments

    :param num_bins int: Maximum number of bins
    :param bin_identity tensor: 1d tensor containing indices of bins
    :param component_identity tensor: 1d tensor containing indices of components
    :output: [num_bins, num_components]
    """
    if bin_identity.size(0) != component_identity.size(0):
        raise ValueError("Error: IIQ9W, size mismatch")
    # both bin_identity and component_identity can contain only non-negative integers
    # sort both by bin_identity
    sorted_bin_identity, order = torch.sort(bin_identity)
    sorted_component_identity = component_identity[order]

    # fill in bins
    filler = torch.tensor([-1], dtype=torch.long, device=device)
    count = 0
    while sorted_bin_identity.nelement() > 0:
        result = torch.unique_consecutive(
            sorted_bin_identity, return_inverse=True, return_counts=(count == 0)
        )
        if count == 0:
            unique_bins, inverse, frequencies = result
            # -1 means blank
            bins = -torch.ones(
                [num_bins, torch.max(frequencies)], dtype=torch.long, device=device,
            )
        else:
            unique_bins, inverse = result

        shifted_inverse = torch.cat([filler, inverse[:-1]])
        is_first_unique = inverse != shifted_inverse
        bins[unique_bins, count] = sorted_component_identity[is_first_unique]

        sorted_bin_identity = sorted_bin_identity[~is_first_unique]
        sorted_component_identity = sorted_component_identity[~is_first_unique]

        count += 1
    return bins


def clean_bins(bins, debug=False, apply_unique=True):
    """
    Given bins,
    1. replace duplicate elements with -1
    2. removes maximum number of tailing -1s
    3. removes empty rows, or rows with only 1 element (ie. no intersecting lines, or a point on a line)
    4. removes duplicate rows

    :param bins tensor: [num_bins, num_components]
    :param debug bool: whether to return number of empty rows removed and duplicate rows removed
    :output: [num_bins, num_components]
    """
    bins = remove_duplicates(bins)

    # remove extra (-1)
    is_redundant = bins == -1
    num_redundant = int(torch.min(torch.sum(is_redundant, dim=-1)))
    if num_redundant > 0:
        bins = bins[:, :-num_redundant]

        # remove empty rows
        is_redundant = is_redundant[:, :-num_redundant]

    is_redundant = torch.sum(is_redundant, dim=-1) >= (is_redundant.size(-1) - 1)
    if debug:
        num_removed_redundant = torch.sum(is_redundant)

    bins = bins[~is_redundant]

    # remove duplicate rows
    if apply_unique:
        bins, inverse = torch.unique(bins, return_inverse=True, dim=0)
        if debug:
            _, counts = torch.unique(inverse, return_counts=True, dim=0)
            num_removed_duplicate = torch.sum(counts - 1)

    ret_vals = [bins]
    if debug:
        ret_vals.append(num_removed_redundant)
    if debug and apply_unique:
        ret_vals.append(num_removed_duplicate)

    # if debug:
    # if apply_unique:
    # num_removed_duplicate = old_bins.size(0) - bins.size(0)
    # return bins, num_removed_redundant, num_removed_duplicate
    # else:
    # return bins, num_removed_redundant
    # return bins
    if len(ret_vals) == 1:
        return ret_vals[0]
    return ret_vals


def remove_duplicates(bins):
    # replace duplicate elements with -1
    # (-2) because (-1) is already used
    bins = torch.sort(bins, descending=True, dim=-1)[0]
    filler = torch.tensor([-2], dtype=torch.long, device=device).expand(bins.size(0), 1)
    shifted_bins = torch.cat([filler, bins[:, :-1]], dim=-1)
    is_duplicate = bins == shifted_bins
    bins[is_duplicate] = -1
    bins = torch.sort(bins, descending=True, dim=-1)[0]

    return bins


def fill_bins(components, bins, filler=-1):
    """
    Given bins of an arbitrary shape and components of an arbitrary shape, replace each index within the bin with the referred component
    An index of (-1) is a filler, and does not refer to any segments

    :param components tensor: [num_components, *]
    :param bins tensor: [num_bins, *]
    :param filler float: value for filler components
    """
    if torch.max(bins) >= components.size(0):
        raise ValueError("Error: LLGUM, indexing error")
    # filled_bins shape: [num_bins, num_components, *component_dim]
    bins_shape = bins.shape
    bins = bins.view(-1)
    is_essential = (bins != -1).view(-1, *[1 for _ in components.shape[1:]])
    # all redundant will have 0 as points
    filled_bins = components[bins] * is_essential + (~is_essential) * filler
    filled_bins = filled_bins.view(*bins_shape, *components.shape[1:])
    return filled_bins


# adapted from https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
# and https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def perp(a):
    # get a vector perpendicular to a
    b = torch.empty_like(a)
    b[..., 0] = -a[..., 1]
    b[..., 1] = a[..., 0]
    return b


def intersect(bins, segments, epsilon=1e-5):
    """
    Given bins and segments, calculate all intersections between segments within each bin, return valid intersections coordinates and the indices of the segments that intersect
    A valid intersection between 2 segments means:
    1. neither identities are (-1)
    2. identities identical (ie. a line against itself)
    3. intersection happens within bounds (ie. doesn't extend) with tolerance epsilon

    If 2 lines are colinear, returns the middle of the borders (see implementation)

    An index of (-1) is a filler, and does not refer to any segments

    :param bins tensor: [num_bins, num_segments]
    :param segments tensor: [num_segments, 2 points, xy coordinates]
    :param epsilon float: error tolerance
    :output segment_identities tensor: [num_intersections, 2 identities]
    :output intersections tensor: [num_intersections, xy coordinates]
    """
    epsilon = torch.as_tensor(epsilon, dtype=torch.double, device=device)

    shape = bins.shape
    # bins_grid: [num_bins, num_segments, num_segments, 2]
    bins_grid = torch.stack(
        [
            bins.unsqueeze(1).expand(shape[0], shape[1], shape[1]),
            bins.unsqueeze(2).expand(shape[0], shape[1], shape[1]),
        ],
        dim=3,
    )
    # preliminary filtering
    # [num_bins, num_segments, num_segments]
    is_essential = torch.all(bins_grid != -1, dim=-1)
    is_unique = bins_grid[..., 0] != bins_grid[..., 1]
    bins = bins_grid[is_essential & is_unique]
    # extract unique comparisons
    bins = torch.sort(bins, dim=-1)[0]
    bins = torch.unique(bins, dim=0)
    # [num_comparisons, 2 lines, 2 points, 2 coordinates]
    filled_bins = fill_bins(segments, bins)
    # [num_comparisons, coords]
    da = filled_bins[:, 0, 1, :] - filled_bins[:, 0, 0, :]
    db = filled_bins[:, 1, 1, :] - filled_bins[:, 1, 0, :]
    dp = filled_bins[:, 0, 0, :] - filled_bins[:, 1, 0, :]
    dap = perp(da)

    # [num_bins, num_segments, num_segments]
    denom = torch.sum(dap * db, dim=-1)
    num = torch.sum(dap * dp, dim=-1)
    # [num_bins, num_segments, num_segments, 2]
    intersections = (num / denom).unsqueeze(-1) * db + filled_bins[:, 1, 0, :]

    # intersection must be within bounds
    # filled_bin_grid shape: [num_comparisons, 2 lines, 2 points, 2 coordinates] - [num_comparisons, 2 lines, 2 coordinates] -> [num_comparisons, 2 coordinates] -> [num_comparisons, 2 coordinates, [lower, upper]]
    borders = torch.stack(
        [
            # lower bound
            torch.max(torch.min(filled_bins, dim=2)[0], dim=1)[0],
            # upper bound
            torch.min(torch.max(filled_bins, dim=2)[0], dim=1)[0],
        ],
        dim=-1,
    )
    # check if range exists
    borders_exist = borders[..., 1] + epsilon >= borders[..., 0] - epsilon
    borders_exist = borders_exist[..., 0] & borders_exist[..., 1]
    # NOTE NOTE NOTE double overlap special case
    is_colinear = (num == 0) & (denom == 0)
    # intersection is simply the middle of border
    intersections[is_colinear] = torch.sum(borders[is_colinear], dim=-1) / 2
    # if colinear or if not parallel
    does_intersect = is_colinear | (denom != 0)

    # check if intersection is within range
    within_borders = (intersections + epsilon >= borders[..., 0] - epsilon) & (
        intersections - epsilon <= borders[..., 1] + epsilon
    )
    within_borders = within_borders[..., 0] & within_borders[..., 1]

    is_valid = borders_exist & within_borders & does_intersect

    segment_identities = bins[is_valid]
    intersections = intersections[is_valid]
    return segment_identities, intersections


# https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
def circle_intersect(max_radius, segments, epsilon=1e-5):
    """
    [TODO:description]

    :param max_radius [TODO:type]: [TODO:description]
    :param segments [TODO:type]: [TODO:description]
    :param epsilon [TODO:type]: [TODO:description]
    :output segment_identities tensor: 1d tensor
    """
    max_radius = torch.as_tensor(max_radius, dtype=torch.double, device=device)
    epsilon = torch.as_tensor(epsilon, dtype=torch.double, device=device)

    segment_identities = (
        torch.arange(segments.size(0), dtype=torch.long, device=device)
        .unsqueeze(1)
        .expand(-1, 2)
    )

    r = max_radius  # Radius of circle
    # [num_segments, xy]
    P1 = segments[:, 0]  # Start of line segment
    # [num_segments, xy]
    V = segments[:, 1] - P1  # Vector along line segment

    # [num_segments]
    a = torch.sum(V * V, dim=-1)
    # [num_segments]
    b = 2 * torch.sum(V * P1, dim=-1)
    # [num_segments]
    c = torch.sum(P1 * P1, dim=-1) - r ** 2

    # discriminant
    # [num_segments]
    disc = b ** 2 - 4 * a * c
    # eventually
    # [num_segments, 2]
    does_intersect = (disc >= 0).unsqueeze(1).expand(-1, 2)

    # [num_segments]
    sqrt_disc = torch.sqrt(disc)
    # [num_segments, 2]
    t = torch.stack([(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)], dim=-1)
    # [num_segments, 2]
    within_borders = ((0 - epsilon) <= t) & (t <= (1 + epsilon))

    # [num_segments, 2, xy]
    intersections = P1.unsqueeze(1) + t.unsqueeze(2) * V.unsqueeze(1)

    is_valid = does_intersect & within_borders

    # not pairs, single
    segment_identities = segment_identities[is_valid]
    intersections = intersections[is_valid]
    return segment_identities, intersections


def connect_circle(num_total_segments, max_radius, segment_identities, intersections):
    # # NOTE: Assuming that 1 hexagon has no shift at ALL
    # num_total_segments is all before non-circle
    if intersections.size(0) <= 1:
        raise ValueError("Error: MYQ7O, not prepared")
    max_radius = torch.as_tensor(max_radius, dtype=torch.double, device=device)
    # [2 segments, 2 coordinates]
    reference_points = torch.zeros([2, 2], dtype=torch.double, device=device)
    reference_points[0, 1] = max_radius
    reference_points[1, 1] = -max_radius
    reference_segment_identities = torch.tensor(
        [num_total_segments, num_total_segments + 1], dtype=torch.long, device=device
    )
    # [2 segments, 2 points, 2 coordinates]
    reference_segments = reference_points.unsqueeze(1).expand(2, 2, 2)
    intersections = torch.cat([reference_points, intersections], dim=0)
    segment_identities = torch.cat(
        [reference_segment_identities, segment_identities], dim=0
    )

    # because you added references
    num_total_segments = num_total_segments + 2

    o = intersections[:, 1]
    a = intersections[:, 0]

    angles = torch.atan2(o, a)
    sorted_angles, order = torch.sort(angles)

    sorted_segment_identities = segment_identities[order]
    sorted_intersections = intersections[order]

    left_segment_identities = (
        torch.remainder(
            torch.arange(sorted_intersections.size(0), dtype=torch.long, device=device)
            * 2
            - 1,
            sorted_intersections.size(0) * 2,
        )
        + num_total_segments
    )
    right_segment_identities = (
        torch.remainder(
            torch.arange(sorted_intersections.size(0), dtype=torch.long, device=device)
            * 2,
            sorted_intersections.size(0) * 2,
        )
        + num_total_segments
    )
    left_right_segment_identities = (
        torch.remainder(
            torch.arange(
                sorted_intersections.size(0) * 2, dtype=torch.long, device=device
            ).view(-1, 2)
            - 1,
            sorted_intersections.size(0) * 2,
        )
        + num_total_segments
    )
    left_segment_identities = torch.stack(
        [left_segment_identities, sorted_segment_identities], dim=-1
    )
    right_segment_identities = torch.stack(
        [right_segment_identities, sorted_segment_identities], dim=-1
    )
    left_intersections = sorted_intersections
    right_intersections = sorted_intersections
    left_right_intersections = sorted_intersections

    # shifted to the left: [0,1,2] -> [1,2,0]
    shifted_segment_identities = torch.roll(sorted_segment_identities, -1, dims=0)
    shifted_intersections = torch.roll(sorted_intersections, -1, dims=0)
    shifted_angles = torch.roll(sorted_angles, -1, dims=0)

    delta_angle = shifted_angles - sorted_angles
    # handle negative delta
    delta_angle = torch.remainder(delta_angle, np.pi * 2)

    # [num_segments]
    # s=theta*r
    length = delta_angle * max_radius
    # stacked_segment_identities = torch.stack(
    # [sorted_segment_identities, shifted_segment_identities], dim=-1
    # )
    # [num_segments, STACKED, xy coordinates]
    stacked_intersections = torch.stack(
        [sorted_intersections, shifted_intersections], dim=1
    )

    # in between 2 consecutive intersections
    # [num_segments, xy]
    midpoint = torch.sum(stacked_intersections, dim=1) / 2
    # between 2 consecutive intersections
    # [num_segments, xy]
    delta_intersections = (
        stacked_intersections[:, 0, :] - stacked_intersections[:, 1, :]
    )
    distance = delta_intersections ** 2
    # [num_segments]
    distance = torch.sqrt(torch.sum(distance, dim=-1))
    max_error = torch.clamp(torch.max(distance - length), min=0)
    print(f"6VWTO << is better: {max_error}")
    length[distance > length] = length[distance > length] + max_error
    if torch.any(distance > length):
        print("Error: 6VWTO")
        print("something wrong with trigonometry")

    offset = (torch.sqrt(length ** 2 - distance ** 2) / 2).unsqueeze(1)
    perp_vector_a = F.normalize(perp(delta_intersections))
    perp_vector_b = -perp_vector_a

    perp_vector_a = perp_vector_a * offset
    perp_vector_b = perp_vector_b * offset

    # [num_segments, 2 prob, xy coords]
    perp_vectors = torch.stack([perp_vector_a, perp_vector_b], dim=1)
    midpoint = midpoint.unsqueeze(1)

    # [num_segments, 2 prob, xy coords]
    offset_intersections = midpoint + perp_vectors

    # select the one farthest from the center
    select = torch.sum(offset_intersections ** 2, dim=-1)
    select = torch.argmax(select, dim=-1)
    # [num_segments, xy coords]
    offset_intersections = offset_intersections[
        torch.arange(select.size(0), dtype=torch.long, device=device), select
    ]
    # + num_total_segments because it will be concatenated to original segments
    offset_segment_identities = (
        torch.arange(
            offset_intersections.size(0) * 2, dtype=torch.long, device=device
        ).view(-1, 2)
        + num_total_segments
    )

    # additional segments to be concatenated, those that form the circle
    # intersections: [num_segments, STACKED, xy coordinates]
    # [num_segments, 2 intersections, xy coords]
    # circle_segments: [num_segments, 2 intersections (left-right), 2 points, xy coords]
    circle_segments = torch.stack(
        [stacked_intersections, offset_intersections.unsqueeze(1).expand(-1, 2, -1)],
        dim=2,
    )
    # [num_segments, 2 points, xy coords]
    circle_segments = circle_segments.view(-1, 2, 2)
    circle_segments = torch.cat([reference_segments, circle_segments], dim=0)

    final_segment_identities = torch.cat(
        [
            left_segment_identities,
            right_segment_identities,
            left_right_segment_identities,
            offset_segment_identities,
        ],
        dim=0,
    )
    final_intersections = torch.cat(
        [
            left_intersections,
            right_intersections,
            left_right_intersections,
            offset_intersections,
        ],
        dim=0,
    )
    # return sorted_segment_identities, sorted_intersections
    return (
        circle_segments,
        final_segment_identities,
        final_intersections,
        reference_segment_identities,
    )


def pad(bins, size):
    """
    Right pad columns, assuming 2d

    :param bins [TODO:type]: [TODO:description]
    :param size [TODO:type]: [TODO:description]
    """
    if bins.size(1) == size:
        return bins
    else:
        if bins.size(1) > size:
            raise ValueError("Error: VRBL8, padding down doesn't make sense")
        pad_size = size - bins.size(1)

        filler = torch.tensor([-1], dtype=torch.long, device=device)
        filler = filler.expand(bins.size(0), pad_size)
        bins = torch.cat([bins, filler], dim=1)
        return bins


def layer_intersections(
    num_segments, segment_identities, intersections, layer_indices, num_layers
):
    """
    Extend intersections to 3d, taking into account layer_indices

    Do nothing for intersections that are within the same layer

    For intersections with segments a-b, in different layers, split it up into a-c and b-c
    Add third coordinate to intersections that indicate layer_indices

    :param segment_identities [TODO:type]: [TODO:description]
    :param intersections [TODO:type]: [TODO:description]
    :param layer_indices [TODO:type]: [TODO:description]
    :param num_layers [TODO:type]: [TODO:description]
    """
    num_intersections = intersections.size(0)
    # [num_intersections, num_segments]
    segment_layer_identities = fill_bins(layer_indices, segment_identities, filler=-1)

    # [num_layers, num_intersections, num_segments]
    on_layer = torch.stack(
        [segment_layer_identities == i for i in range(num_layers)], dim=0
    )
    repeat = torch.sum(on_layer, dim=-1)

    per_layer_bin_identity = torch.arange(
        intersections.size(0), dtype=torch.long, device=device
    )
    per_layer_bin_identity = [
        per_layer_bin_identity.repeat_interleave(repeat[i]) for i in range(num_layers)
    ]
    # intersections.

    # segments on each intersection that match a layer
    # [num_layers, num_chosen]
    per_layer_component = [segment_identities[on_layer[i]] for i in range(num_layers)]

    # [num_layers, num_segments, num_chosen]
    per_layer_bins = [
        binning(
            intersections.size(0), per_layer_bin_identity[i], per_layer_component[i]
        )
        for i in range(num_layers)
    ]
    max_num_chosen = max([i.size(1) for i in per_layer_bins])
    # [num_layers, intersections, num_chosen]
    per_layer_bins = torch.stack(
        [pad(i, max_num_chosen) for i in per_layer_bins], dim=0
    )

    # [num_intersections]
    is_same = torch.sum(per_layer_bins[:, :, 0] != -1, dim=0)

    if torch.any(is_same == 0):
        raise ValueError("Error: L0PM5, why are some blank")

    # intersections on the same layer only 1 nonzero
    is_same = is_same == 1
    # [num_same, num_chosen]
    # WELWP, this might be faulty
    same_segment_identities = segment_identities[is_same]
    same_segment_layer_identities = segment_layer_identities[is_same][:, 0].unsqueeze(1)
    # [num_same, 3]
    same_intersections = torch.cat(
        [intersections[is_same], same_segment_layer_identities], dim=1
    )

    # monkey patch 3d vertical segments
    num_diff = torch.sum(~is_same)
    if num_diff:
        # [layers, intersections, num_chosen]
        diff_segment_identities = per_layer_bins[:, ~is_same]
        diff_concat = (
            torch.arange(
                num_segments, num_segments + num_diff, dtype=torch.long, device=device
            )
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(num_layers, num_diff, 1)
        )
        # check if the first isn't -1
        is_valid = diff_segment_identities[:, :, 0] != -1
        diff_segment_identities = torch.cat(
            [diff_concat, diff_segment_identities], dim=-1
        )

        diff_segment_layer_identities = (
            torch.arange(num_layers, dtype=torch.long, device=device)
            .unsqueeze(1)
            .unsqueeze(-1)
            .expand(num_layers, num_diff, 1)
        )
        diff_intersections = torch.cat(
            [
                intersections[~is_same].unsqueeze(0).expand(num_layers, num_diff, 2),
                diff_segment_layer_identities,
            ],
            dim=-1,
        )
        # NOTE: the endpoints don't necessarily contain the proper layer coordinate. Too much of a hassle
        # [num_segments, 2, 3 coords]
        vertical_segments = torch.stack(
            [diff_intersections[0], diff_intersections[-1]], dim=1
        )

        # [num_diff, num_chosen]
        diff_segment_identities = diff_segment_identities[is_valid]
        # [num_diff, 3]
        diff_intersections = diff_intersections[is_valid]

    # consolidate different num_chosen in same and diff
    # WEOWEWOEWOEWOEWOEWOOWOEOWEWOEOWEWEOWOEOEWEOW
    same_segment_identities, num_removed_redundant = clean_bins(
        same_segment_identities, debug=True, apply_unique=False
    )
    num_removed_duplicate = same_segment_identities.size(0) - torch.unique(
        same_segment_identities, dim=0
    ).size(0)
    if num_removed_duplicate or num_removed_redundant:
        raise ValueError("Error: FR61V")

    if num_diff:
        diff_segment_identities, num_removed_redundant = clean_bins(
            diff_segment_identities, debug=True, apply_unique=False
        )
        num_removed_duplicate = diff_segment_identities.size(0) - torch.unique(
            diff_segment_identities, dim=0
        ).size(0)
        if num_removed_duplicate or num_removed_redundant:
            raise ValueError("Error: H4CNV")

        max_num_chosen = max(
            same_segment_identities.size(1), diff_segment_identities.size(1)
        )
        same_segment_identities = pad(same_segment_identities, max_num_chosen)
        diff_segment_identities = pad(diff_segment_identities, max_num_chosen)

        # [num_intersections, num_chosen]
        final_segment_identities = torch.cat(
            [same_segment_identities, diff_segment_identities], dim=0
        )
        # [num_intersections, 3]
        final_intersections = torch.cat([same_intersections, diff_intersections], dim=0)

        return vertical_segments, final_segment_identities, final_intersections
    return None, same_segment_identities, same_intersections


def connect(
    max_radius,
    num_segments,
    segment_identities,
    intersections,
    reference_segment_identities,
    epsilon,
):
    """
    Given intersections and indices of intersecting segments, generate a matrix for the node-voltage equation (see https://en.wikipedia.org/wiki/Nodal_analysis), assuming a line of length 1 has a resistance of 1 ohm

    Also takes into account intersections with identical coordinates, to prevent infinite conductance

    Output matrix is in COO format, because sparsity :(

    Also adds interlayer resistance

    :param num_segments int: Maximum number of segments
    :param num_vertical int: Number of vertical line segments, use to exclude it from clean_bins
    :param circle tuple(segment_identities, intersections):
    :output segment_identities tensor: [num_intersections, 2 identities]
    :output intersections tensor: [num_intersections, xy coordinates]
    """

    num_chosen = segment_identities.size(1)
    intersect_identities = torch.arange(
        intersections.size(0), dtype=torch.long, device=device
    ).repeat_interleave(num_chosen)

    is_redundant = segment_identities.view(-1) == -1

    # bins = segments, components = intersections on segment
    bins = binning(
        num_segments,
        segment_identities.view(-1)[~is_redundant],
        intersect_identities[~is_redundant],
    )
    bins, num_removed_redundant, num_removed_duplicate = clean_bins(bins, debug=True)
    if num_removed_duplicate:
        raise ValueError("Error: NCXV6")
    print(f"{num_removed_redundant} segments orphaned")
    unique_vals = torch.unique(bins.view(-1))
    unique_vals = unique_vals[unique_vals != -1]

    if unique_vals.size(0) != intersections.size(0):
        raise ValueError("Error: W7BH9, orphaned intersection, case not handled: just remove intersection")

    bins, distance = sort_occurences(intersections, bins)
    is_duplicate = (distance <= epsilon) & (distance > -1)
    if torch.any(is_duplicate):
        raise ValueError("Error: JNLCL, merge failed")

    conductance = 1 / distance
    if torch.any(torch.isinf(conductance)):
        raise ValueError("Error: NM30A, infinite conductance")

    # # generate COO format
    row_indices = bins[:, :-1]
    col_indices = bins[:, 1:]
    connect_indices = torch.stack([row_indices, col_indices], dim=-1).view(-1, 2)
    connect_indices = torch.sort(connect_indices, dim=-1)[0]
    conductance = conductance.view(-1)

    is_redundant = torch.any(connect_indices == -1, dim=-1)
    connect_indices = connect_indices[~is_redundant]
    conductance = conductance[~is_redundant]

    # just double checking
    if torch.any(connect_indices[:, 0] == connect_indices[:, 1]):
        raise ValueError("Error: CP77K, self reference")

    naive_combine = torch.cat(
        [connect_indices.double(), conductance.unsqueeze(-1)], dim=-1
    )
    unique_combine = torch.unique(naive_combine, dim=0)

    if unique_combine.size(0) != torch.unique(connect_indices, dim=0).size(0):
        raise ValueError("Error: 8LUPN, duplicate in COO")

    connect_indices = unique_combine[:, :-1].long()
    conductance = unique_combine[:, -1]

    # mirror
    connect_indices = torch.cat(
        [connect_indices, torch.flip(connect_indices, dims=(1,))], dim=0
    )
    # repeat NOT repeat_interleave
    conductance = conductance.repeat(2)
    num_bins = intersections.size(0)
    # calculate diagonal
    diagonal_bins = binning(
        num_bins,
        connect_indices[:, 0],
        torch.arange(conductance.size(0), dtype=torch.long, device=device),
    )
    filled_diagonal_bins = fill_bins(conductance, diagonal_bins, filler=0)
    diagonal_conductance = torch.sum(filled_diagonal_bins, dim=-1)
    diagonal_connect_indices = (
        torch.arange(num_bins, dtype=torch.long, device=device)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    if torch.sum(diagonal_conductance == 0) != 0:
        raise ValueError("Error: TDJU4, zero conductance on node")

    # change the sign of normal conductance
    conductance = -conductance

    conductance = torch.cat([conductance, diagonal_conductance], dim=0)
    connect_indices = torch.cat([connect_indices, diagonal_connect_indices], dim=0)

    # row-major sorting: if not, it will print errors
    final_connect_indices, inverse = torch.unique(
        connect_indices, sorted=True, dim=0, return_inverse=True
    )
    final_conductance = torch.zeros_like(conductance)
    final_conductance[inverse] = conductance

    if final_connect_indices.size(0) != connect_indices.size(0):
        raise ValueError("Error: 57ZDU, size mismatch")

    reference_intersection_a = torch.nonzero(
        torch.any(segment_identities == reference_segment_identities[0], dim=-1),
        as_tuple=False,
    ).squeeze(-1)
    reference_intersection_b = torch.nonzero(
        torch.any(segment_identities == reference_segment_identities[1], dim=-1),
        as_tuple=False,
    ).squeeze(-1)

    if (reference_intersection_a.size(0) != 1) or (
        reference_intersection_b.size(0) != 1
    ):
        raise ValueError("Error: SL002, why are there multiple references")
    reference_intersection_a = reference_intersection_a[0]
    reference_intersection_b = reference_intersection_b[0]

    # (Electric) Current Vector
    current = torch.zeros(num_bins, dtype=torch.double, device=device)
    # NOTE: set input, output current here
    current[reference_intersection_a] = 1
    current[reference_intersection_b] = -1
    # print, might need to delete last row and column, to prevent singular matrix

    # remove last row & column
    is_last = torch.any(
        final_connect_indices == torch.max(final_connect_indices), dim=-1
    )
    cut_connect_indices = final_connect_indices[~is_last]
    cut_conductance = final_conductance[~is_last]
    cut_current = current[:-1]

    solve_params = {
        "size": num_bins - 1,
        "row": cut_connect_indices[:, 0],
        "col": cut_connect_indices[:, 1],
        "data": cut_conductance,
        "b": cut_current,
        "reference_intersections": torch.stack(
            [reference_intersection_a, reference_intersection_b]
        ),
    }
    return (segment_identities, intersections, solve_params)


def merge(max_radius, segment_identities, intersections, epsilon):
    # is_circle = circle_identities is not None
    epsilon = torch.as_tensor(epsilon, dtype=torch.double, device=device)

    # putting intersections in geometric bins
    # [num_bins, intersections]
    intersection_bin = generate_bins(intersections, max_radius)
    # [num_bins, intersections, coords]
    filled_intersection_bin = fill_bins(intersections, intersection_bin)

    shape = intersection_bin.shape
    # bins_grid: [num_bins, num_segments, num_segments, 2]
    intersection_bin_grid = torch.stack(
        [
            intersection_bin.unsqueeze(1).expand(shape[0], shape[1], shape[1]),
            intersection_bin.unsqueeze(2).expand(shape[0], shape[1], shape[1]),
        ],
        dim=3,
    )
    is_valid = torch.all(intersection_bin_grid != -1, dim=-1)

    shape = filled_intersection_bin.shape
    # bins_grid: [num_bins, num_segments, num_segments, 2, 2 coords]
    filled_intersection_bin_grid = torch.stack(
        [
            filled_intersection_bin.unsqueeze(1).expand(
                shape[0], shape[1], shape[1], shape[2]
            ),
            filled_intersection_bin.unsqueeze(2).expand(
                shape[0], shape[1], shape[1], shape[2]
            ),
        ],
        dim=3,
    )

    distance = torch.sqrt(
        torch.sum(
            (
                filled_intersection_bin_grid[..., 0, :]
                - filled_intersection_bin_grid[..., 1, :]
            )
            ** 2,
            dim=-1,
        )
    )
    distance = (distance * is_valid) - (~is_valid).double()
    # and not on the diagonal
    is_duplicate = (distance <= epsilon) & (distance > -1)
    merge_source = intersection_bin_grid[is_duplicate]
    # remove diagonals
    merge_source = merge_source[merge_source[:, 0] != merge_source[:, 1]]
    merge_source = torch.sort(merge_source, dim=-1)[0]
    merge_source = torch.unique(merge_source, dim=0)

    # based on INTERSECTIONS, not segments
    # Yes, some are redundant, ie. [728, 728], but it doesn't matter
    merge_target = merge_duplicate(merge_source)

    merge_source = merge_source.view(-1)
    merge_target = merge_target.view(-1)

    merge_mapping = torch.stack([merge_source, merge_target], dim=-1)
    merge_mapping = torch.unique(merge_mapping, dim=0)  # , sorted=True) not necessary
    # check if duplicate source
    if torch.unique(merge_mapping[:, 0]).size(0) != merge_mapping.size(0):
        raise ValueError("Error: 63K5C, problem with unique")
    # use bin to index this tensor to obtain new indices
    merge_indices = torch.arange(intersections.size(0), dtype=torch.long, device=device)
    # replace source with target
    merge_indices[merge_mapping[:, 0]] = merge_mapping[:, 1]

    # # (-1) IS A MUST, to keep redundancy
    # bins = fill_bins(merge_indices, bins, filler=-1)

    # do the same to segment_identities
    # bins = new_order from merge_indices, component = arange
    segment_bins = binning(
        intersections.size(0),
        merge_indices,
        # should this be arange?
        torch.arange(intersections.size(0), dtype=torch.long, device=device),
    )
    segment_identities = fill_bins(segment_identities, segment_bins, filler=-1)
    segment_identities = segment_identities.view(segment_identities.size(0), -1)
    # reduce maximum index
    reduce_source = torch.sort(merge_indices)[0]
    reduce_source = torch.unique_consecutive(reduce_source)
    segment_identities = segment_identities[reduce_source]
    # segment_identities = remove_duplicates(segment_identities)
    # DO NOT APPLY unique, or else order will get scrambled
    segment_identities, num_removed_redundant = clean_bins(
        segment_identities, debug=True, apply_unique=False
    )
    num_removed_duplicate = segment_identities.size(0) - torch.unique(
        segment_identities, dim=0
    ).size(0)
    if num_removed_duplicate or num_removed_redundant:
        raise ValueError("Error: F6X57, segment row deleted")

    # corresponds to segment_identities
    intersections = intersections[reduce_source]
    if intersections.size(0) != segment_identities.size(0):
        raise ValueError("Error: RIDVK, segment_identities intersections mismatch")

    return segment_identities, intersections


def merge_reduce(
    bins,
    distance,
    segment_identities,
    intersections,
    # circle_identities=None,
    epsilon=1e-10,
):
    """
    Recalculate segment_identities and intersections, to contain lowest possible number of rows, by compressing via merge_duplicate
    i.e. if segment_identities = [[2, 3], [4, 5]] and the merge_target is [0, 0], new_segment_identities = [[2, 3, 4, 5]]
    if bins = [[1,2,3,4,5]], new_bins = [[0,1,2,3,4]], whilst rearranging intersections so this remains true

    do everything the same to circle_identities as you did to segment_identities

    :param bins [TODO:type]: [TODO:description]
    :param distance [TODO:type]: [TODO:description]
    :param segment_identities [TODO:type]: [TODO:description]
    :param intersections [TODO:type]: [TODO:description]
    """

    # is_circle = circle_identities is not None
    epsilon = torch.as_tensor(epsilon, dtype=torch.double, device=device)

    row_indices = bins[:, :-1]
    col_indices = bins[:, 1:]

    # Merge distances under epsilon, but also filter -1s
    # is_duplicate = distance == 0
    is_duplicate = (distance <= epsilon) & (distance > -1)
    if torch.sum(is_duplicate) == 0:
        raise ValueError("Error: I51WK, case not handled")
    merge_source = torch.stack([row_indices, col_indices], dim=-1)[is_duplicate]
    # based on INTERSECTIONS, not segments
    # Yes, some are redundant, ie. [728, 728], but it doesn't matter
    merge_target = merge_duplicate(merge_source)

    merge_source = merge_source.view(-1)
    merge_target = merge_target.view(-1)

    merge_mapping = torch.stack([merge_source, merge_target], dim=-1)
    merge_mapping = torch.unique(merge_mapping, dim=0)  # , sorted=True) not necessary
    # check if duplicate source
    if torch.unique(merge_mapping[:, 0]).size(0) != merge_mapping.size(0):
        raise ValueError("Error: 63K5C, problem with unique")
    # use bin to index this tensor to obtain new indices
    merge_indices = torch.arange(intersections.size(0), dtype=torch.long, device=device)
    # replace source with target
    merge_indices[merge_mapping[:, 0]] = merge_mapping[:, 1]
    # (-1) IS A MUST, to keep redundancy
    bins = fill_bins(merge_indices, bins, filler=-1)

    # do the same to segment_identities
    # bins = new_order from merge_indices, component = arange
    segment_bins = binning(
        intersections.size(0),
        merge_indices,
        # should this be arange?
        torch.arange(intersections.size(0), dtype=torch.long, device=device),
    )
    segment_identities = fill_bins(segment_identities, segment_bins, filler=-1)
    segment_identities = segment_identities.view(segment_identities.size(0), -1)
    # reduce maximum index
    reduce_source = torch.sort(merge_indices)[0]
    reduce_source = torch.unique_consecutive(reduce_source)
    segment_identities = segment_identities[reduce_source]
    # segment_identities = remove_duplicates(segment_identities)
    segment_identities, num_removed_redundant = clean_bins(
        segment_identities, debug=True, apply_unique=False
    )
    num_removed_duplicate = segment_identities.size(0) - torch.unique(
        segment_identities, dim=0
    ).size(0)
    if num_removed_duplicate or num_removed_redundant:
        raise ValueError("Error: F6X57, segment row deleted")

    # corresponds to segment_identities
    intersections = intersections[reduce_source]
    if intersections.size(0) != segment_identities.size(0):
        raise ValueError("Error: RIDVK, segment_identities intersections mismatch")

    reduce_target = torch.arange(reduce_source.size(0), dtype=torch.long, device=device)
    reduce_indices = -2 * torch.ones(
        torch.max(reduce_source) + 1, dtype=torch.long, device=device
    )
    # NOTE:  MIGHT BE FLIPPED
    reduce_indices[reduce_source] = reduce_target

    # (-1) IS A MUST, to keep redundancy
    bins = fill_bins(reduce_indices, bins, filler=-1)
    if torch.any(bins == -2):
        raise ValueError("Error: NT9OF")
    return bins, segment_identities, intersections


def sort_occurences(intersections, bins):
    """
    Given bins (with each bin being a segment, containing intersections on that point), sort the intersections points, so that the distance between intersection 0 and intersection n (for n>0) keeps increasing for larger n. Also returns distance between consecutive elements

    An index of (-1) is a filler, and does not refer to any segments

    :param intersections tensor: [num_intersections, xy coordinates]
    :param bins tensor: [num_segments, num_intersections]
     output bins tensor: [num_segments, num_intersections]
    :output distance tensor: [num_segments, num_intersections-1]
    """
    filled_bins = fill_bins(intersections, bins)

    """
    Get outermost reference point by
    1. choose an arbitrary point on the segment (ie. first index)
    2. measuring distance of all points to that point
    3. choose the point with the highest distance as the reference point
    4. sort all points according to this reference point
    """
    # (0:1) is just to keep dimensions
    distance = filled_bins - filled_bins[:, 0:1, :]
    # no need for sqrt, since it doesn't change order
    distance = torch.sum(distance ** 2, dim=-1)
    distance[bins == -1] = -1

    values, reference = torch.max(distance, dim=-1)
    if torch.any(values < 0):
        raise ValueError("Error: J93KI")

    reference = filled_bins[torch.arange(filled_bins.size(0)), reference, :]

    distance = filled_bins - reference.unsqueeze(1)
    distance = torch.sum(distance ** 2, dim=-1)
    distance[bins == -1] = -1

    # sort identities according to distance to reference
    sorted_distances, order = torch.sort(distance, descending=True, dim=-1)
    bins = torch.gather(bins, 1, order)
    filled_bins = fill_bins(intersections, bins)
    # NOTE: handle symmetry (row_indices and col_indices are interchangeable)
    # NOTE: handle intersections that were discarded during clean bin

    if bins.size(1) < 2:
        raise ValueError("Error: HMLSE")

    # distances between consecutive points
    distance = (filled_bins[:, 1:] - filled_bins[:, :-1]) ** 2
    distance = torch.sqrt(torch.sum(distance, dim=-1))
    is_redundant = (bins[:, 1:] == -1) | (bins[:, :-1] == -1)
    distance[is_redundant] = -1

    return bins, distance


def merge_duplicate(merge_source):
    """
    Given a list of pairs of indices, return the highest index that has been paired up with that index, and that index's pair, so on
    for example,

    merge_source = [
    [0, 1],
    [3, 0],
    ]
    merge_target = [
    [3, 3],
    [3, 3],
    ]

    merge_source = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    ]
    merge_target = [
    [4, 4],
    [4, 4],
    [4, 4],
    [4, 4],
    ]

    :param merge_source tensor: [num_pairs, 2]
    :output tensor: [num_paris, 2]
    """
    # merge_source refers to the intersection identity of duplicates
    # target refers to what to map it to
    merge_target = torch.max(merge_source, dim=-1)[0].repeat_interleave(2)
    merge_source = merge_source.view(-1)

    # just for debugging
    count = 0
    sorted_merge_source, order = torch.sort(merge_source)
    while True:
        sorted_merge_target = merge_target[order]

        # use inverse for binning because gaps in merge_source (ie. intersections which aren't duplicate)
        unique_merge_source, inverse, frequencies = torch.unique_consecutive(
            sorted_merge_source, return_inverse=True, return_counts=True
        )
        # bins = intersection identitites, components = merge index
        merge_bins = binning(unique_merge_source.size(0), inverse, sorted_merge_target)
        # corresponds to unique_merge_source
        new_merge_target = torch.max(merge_bins, dim=-1)[0]
        # undo unique
        # corresponds sorted_merge_source
        non_unique_merge_target = torch.repeat_interleave(new_merge_target, frequencies)
        # undo sort
        """
        explanation
        a = torch.tensor([0,3,4,2])
        b,c = torch.sort(a)
        b==torch.tensor([0,2,3,4])
        c==torch.tensor([0,3,1,2])
        d = torch.zeros_like(b)
        d[c]=b
        then,
        d==a
        """
        unsorted_merge_target = -torch.ones_like(non_unique_merge_target)
        unsorted_merge_target[order] = non_unique_merge_target
        if torch.any(unsorted_merge_target == -1):
            raise ValueError("Error: H1T0Q")

        final_merge_target = unsorted_merge_target.view(-1, 2)
        final_merge_target = torch.max(final_merge_target, dim=-1, keepdim=True)[
            0
        ].expand(-1, 2)
        final_merge_target = final_merge_target.reshape(-1)

        count += 1
        # check if complete
        if torch.all(final_merge_target == merge_target):
            break
        else:
            merge_target = final_merge_target
    print(str(count) + " iterations of merge_duplicate")
    return final_merge_target.view(-1, 2)


def draw(
    max_radius,
    segments,
    intersections=None,
    highlight_segments=None,
    highlight_intersections=None,
    draw_circle=True,
    small_circle=None,
    plot=True,
):
    """
    Draw (and highlight) segments and intersections in Matplotlib, aint with a circle of radius max_radius

    :param max_radius float: radius of circle
    :param segments tensor: [num_segments, 2 points, xy coordinates]
    :param intersections tensor: [num_intersections, xy coordinates]
    :param highlight_segments tensor: 1d indices of segments to highlight
    :param highlight_intersections tensor: 1d indices of intersections to highlight
    """
    print(f"# segments:{len(segments)}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax = plt.gca()
    ax.set_aspect("equal")

    ax.set_xlim(-max_radius * 1.1 - 1, max_radius * 1.1 + 1)
    ax.set_ylim(-max_radius * 1.1 - 1, max_radius * 1.1 + 1)

    collection = LineCollection(segments.cpu(), colors=["b"], zorder=0)
    ax.add_collection(collection)

    if intersections is not None:
        ax.scatter(
            intersections[:, 0].cpu(), intersections[:, 1].cpu(), c=["g"], zorder=2
        )

    if draw_circle:
        circ = plt.Circle((0, 0), radius=max_radius, color="r", fill=False, zorder=1)
        ax.add_patch(circ)
    if small_circle is not None:
        circ = plt.Circle((0, 0), radius=small_circle, color="r", fill=False, zorder=1)
        ax.add_patch(circ)

    if highlight_segments is not None:
        collection = LineCollection(segments[highlight_segments].cpu(), colors=["g"],)
        ax.add_collection(collection)
    if highlight_intersections is not None:
        ax.scatter(
            intersections[highlight_intersections, 0].cpu(),
            intersections[highlight_intersections, 1].cpu(),
            c=["r"],
            zorder=3,
        )
    fig.tight_layout()
    if plot:
        plt.show()


def main(inner_radius, outer_radius, theta, enable_interlayer):
    segments = []
    segments.append(generate_hexagon(outer_radius, theta=-theta / 2))
    segments.append(generate_hexagon(outer_radius, theta=theta / 2))
    num_layers = len(segments)

    # NOTE NOTE NOTE: making a sandwich
    # NOTE *2 because of 1 middle layer (circle)?
    layer_indices = torch.arange(
        num_layers, dtype=torch.long, device=device
    ).repeat_interleave(torch.tensor([i.size(0) for i in segments], device=device))*2

    segments = torch.cat(segments)
    num_segments = segments.size(0)

    circle_identities, circle_intersections = circle_intersect(inner_radius, segments)
    (
        circle_segments,
        circle_segment_identities,
        circle_intersections,
        reference_segment_identities,
    ) = connect_circle(
        num_segments, inner_radius, circle_identities, circle_intersections
    )

    bins = generate_bins(segments, outer_radius)
    segment_identities, intersections = intersect(bins, segments)

    segments = torch.cat([segments, circle_segments], dim=0)
    num_segments = segments.size(0)

    layer_indices = torch.cat(
        [
            layer_indices,
            torch.tensor(
                # no +1, because arange is not inclusive
                # NOTE NOTE NOTE: sandwich
                [1],
                # [num_layers],
                dtype=torch.long,
                device=device,
            ).expand(circle_segments.size(0)),
        ],
        dim=0,
    )
    # NOTE: interlayer resistance toggle
    if enable_interlayer:
        num_layers = num_layers + 1
    else:
        layer_indices = torch.zeros_like(layer_indices)
        num_layers = 1

    segment_identities = torch.cat(
        [segment_identities, circle_segment_identities], dim=0
    )

    intersections = torch.cat([intersections, circle_intersections], dim=0)

    epsilon = 1e-10
    segment_identities, intersections = merge(
        outer_radius, segment_identities, intersections, epsilon=epsilon
    )
    vertical_segments, segment_identities, intersections = layer_intersections(
        num_segments, segment_identities, intersections, layer_indices, num_layers
    )

    num_vertical = vertical_segments.size(0) if vertical_segments is not None else 0
    # draw(outer_radius, segments)
    (segment_identities, intersections, solve_params,) = connect(
        outer_radius,
        segments.size(0) + num_vertical,
        segment_identities,
        intersections,
        reference_segment_identities,
        epsilon,
    )
    torch.cuda.empty_cache()
    dx, resistance = circuit_solve(**solve_params)
    print(f"resistance: {resistance}")
    # draw(outer_radius, segments, intersections)
    # draw(outer_radius, segments)
    # draw(radius, segments, highlight_segments=reference_segment_identities)
    if __name__ == "__main__":
        if max(inner_radius, outer_radius) < draw_threshold:
            draw(
                outer_radius,
                segments,
                intersections,
                highlight_intersections=solve_params["reference_intersections"],
            )
    # draw(outer_radius, segments, intersections[reference_segment_identities])
    # draw(outer_radius, segments, segments[reference_segment_identities,0,:])
    # draw(radius, segments, reference)
    return resistance


# def dummy_bins(segments):
    # # segments with shape [num_segments, points, coordinates]
    # filled_bins = segments.unsqueeze(0)
    # bins = torch.arange(segments.size(0), dtype=torch.long, device=device).unsqueeze(0)
    # return bins, filled_bins


if __name__ == "__main__":
    # debug()
    # Causes error
    # main(2, 2, np.pi)  # 1.1, 1.2
    # 40 is about maximum
    # inner, outer
    main(15, 30, np.pi/2, enable_interlayer=False)  # 1.1, 1.2
    # main(100, 100, 2e-3)  # 1.1, 1.2
