use crate::profile::{CUTOFF_RADIUS, MAX_POTENTIAL};
use crate::Geometry;
use feos_core::EosUnit;
use gauss_quad::GaussLegendre;
use ndarray::{Array1, Array2, Zip};
use ndarray_stats::SummaryStatisticsExt;
use quantity::{QuantityArray2, QuantityScalar};
use std::f64::consts::PI;
use std::usize;

// Calculate free-energy average potential for given solid structure.
pub fn calculate_fea_potential<U: EosUnit>(
    grid: &Array1<f64>,
    mi: f64,
    coordinates: &QuantityArray2<U>,
    sigma_sf: Array1<f64>,
    epsilon_k_sf: Array1<f64>,
    pore_center: &[f64; 3],
    system_size: &[QuantityScalar<U>; 3],
    n_grid: &[usize; 2],
    temperature: f64,
    geometry: Geometry,
    cutoff_radius: Option<f64>,
) -> Array1<f64> {
    // allocate external potential
    let mut potential: Array1<f64> = Array1::zeros(grid.len());

    // calculate squared cutoff radius
    let cutoff_radius2 = cutoff_radius.unwrap_or(CUTOFF_RADIUS).powi(2);

    // dimensionless solid coordinates
    let coordinates = Array2::from_shape_fn(coordinates.raw_dim(), |(i, j)| {
        (coordinates.get((i, j)))
            .to_reduced(U::reference_length())
            .unwrap()
    });

    let system_size = [
        system_size[0].to_reduced(U::reference_length()).unwrap(),
        system_size[1].to_reduced(U::reference_length()).unwrap(),
        system_size[2].to_reduced(U::reference_length()).unwrap(),
    ];

    // Create secondary axis:
    // Cartesian coordinates => y
    // Cylindrical coordinates => phi
    // Spherical coordinates => phi
    let (nodes1, weights1) = match geometry {
        Geometry::Cartesian => {
            let nodes = Array1::linspace(
                0.5 * system_size[1] / n_grid[0] as f64,
                system_size[1] - 0.5 * system_size[1] / n_grid[0] as f64,
                n_grid[0],
            );
            let weights = Array1::from_elem(n_grid[0], system_size[1] / n_grid[0] as f64);
            (nodes, weights)
        }
        Geometry::Spherical | Geometry::Cylindrical => {
            let nodes = PI + Array1::from_vec(GaussLegendre::nodes_and_weights(n_grid[0]).0) * PI;
            let weights = Array1::from_vec(GaussLegendre::nodes_and_weights(n_grid[0]).1) * PI;
            (nodes, weights)
        }
    };

    // Create tertiary axis
    // Cartesian coordinates => z
    // Cylindrical coordinates => z
    // Spherical coordinates => theta
    let (nodes2, weights2) = match geometry {
        Geometry::Cylindrical | Geometry::Cartesian => {
            let nodes = Array1::linspace(
                0.5 * system_size[2] / n_grid[1] as f64,
                system_size[2] - 0.5 * system_size[2] / n_grid[1] as f64,
                n_grid[1],
            );
            let weights = Array1::from_elem(n_grid[1], system_size[2] / n_grid[1] as f64);
            (nodes, weights)
        }
        Geometry::Spherical => {
            let nodes = PI / 2.0
                + Array1::from_vec(GaussLegendre::nodes_and_weights(n_grid[1]).0) * PI / 2.0;
            let weights = Array1::from_vec(GaussLegendre::nodes_and_weights(n_grid[1]).1) * PI
                / 2.0
                * Array1::from_shape_fn(n_grid[1], |i| nodes[i].sin());
            (nodes, weights)
        }
    };

    // calculate weights
    let weights = Array2::from_shape_fn((n_grid[0], n_grid[1]), |(i, j)| weights1[i] * weights2[j]);

    // calculate sum of weights
    let weights_sum = weights.sum();

    // calculate FEA potential
    Zip::indexed(&mut potential).par_for_each(|i0, f| {
        let mut potential_2d: Array2<f64> = Array2::zeros((n_grid[0], n_grid[1]));
        for (i1, &n1) in nodes1.iter().enumerate() {
            for (i2, &n2) in nodes2.iter().enumerate() {
                let point = match geometry {
                    Geometry::Cartesian => [grid[i0], n1, n2],
                    Geometry::Cylindrical => [
                        pore_center[0] + grid[i0] * n1.cos(),
                        pore_center[1] + grid[i0] * n1.sin(),
                        n2,
                    ],
                    Geometry::Spherical => [
                        pore_center[0] + grid[i0] * n2.sin() * n1.cos(),
                        pore_center[1] + grid[i0] * n2.sin() * n1.sin(),
                        pore_center[2] + grid[i0] * n2.cos(),
                    ],
                };

                let distance2 = calculate_distance2(point, &coordinates, system_size);
                let potential_sum: f64 = (0..sigma_sf.len())
                    .map(|alpha| {
                        mi * evaluate(
                            distance2[alpha],
                            sigma_sf[alpha],
                            epsilon_k_sf[alpha],
                            cutoff_radius2,
                        ) / temperature
                    })
                    .sum();
                potential_2d[[i1, i2]] = (-potential_sum.min(MAX_POTENTIAL)).exp();
            }
        }
        *f = potential_2d.weighted_sum(&weights).unwrap();
    });

    -temperature * potential.map(|p| (p / weights_sum).ln())
}

// -temperature * potential.map(|p| (p / weights_sum).ln())

/// Evaluate LJ12-6 potential between solid site "alpha" and fluid segment
fn evaluate(distance2: f64, sigma: f64, epsilon: f64, cutoff_radius2: f64) -> f64 {
    let sigma_r = sigma.powi(2) / distance2;

    let potential: f64 = if distance2 > cutoff_radius2 {
        0.0
    } else if distance2 == 0.0 {
        f64::INFINITY
    } else {
        4.0 * epsilon * (sigma_r.powi(6) - sigma_r.powi(3))
    };

    potential
}

/// Evaluate the squared euclidian distance between a point and the coordinates of all solid atoms.
fn calculate_distance2(
    point: [f64; 3],
    coordinates: &Array2<f64>,
    system_size: [f64; 3],
) -> Array1<f64> {
    Array1::from_shape_fn(coordinates.ncols(), |i| {
        let mut rx = coordinates[[0, i]] - point[0];
        let mut ry = coordinates[[1, i]] - point[1];
        let mut rz = coordinates[[2, i]] - point[2];

        rx = rx - system_size[0] * (rx / system_size[0]).round();
        ry = ry - system_size[1] * (ry / system_size[1]).round();
        rz = rz - system_size[2] * (rz / system_size[2]).round();

        rx.powi(2) + ry.powi(2) + rz.powi(2)
    })
}
