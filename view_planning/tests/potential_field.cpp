#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <deque>
#include <random>
#include <string>
#include <numeric>

#include <open3d/Open3D.h>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkSphereSource.h>
#include <vtkIdTypeArray.h>
#include <vtkPoints.h>

/**
 * STRUCTS
 */
// Hash function for Eigen::Vector3d
// Used for hashing positions in unordered_maps for efficient lookups.
struct Vector3dHash {
    std::size_t operator()(const Eigen::Vector3d& vec) const {
        std::size_t h1 = std::hash<double>{}(vec.x());
        std::size_t h2 = std::hash<double>{}(vec.y());
        std::size_t h3 = std::hash<double>{}(vec.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};


/**
 * UTILITY FUNCTIONS
 */

// Generate viewpoints clustered near a specific region on the sphere.
// Parameters:
// - num_viewpoints: Number of viewpoints to generate.
// - sphere_radius: Radius of the sphere on which viewpoints are placed.
// Returns: Vector of shared pointers to Viewpoint objects.
std::vector<std::shared_ptr<visioncraft::Viewpoint>> generateClusteredViewpoints(int num_viewpoints, float sphere_radius) {
    std::vector<std::shared_ptr<visioncraft::Viewpoint>> viewpoints;

    float central_theta = M_PI / 20;  // Central azimuthal angle
    float central_phi = M_PI / 20;   // Central polar angle
    float spread = M_PI / 20;       // Range of deviation

    for (int i = 0; i < num_viewpoints; ++i) {
        float theta = central_theta + (static_cast<float>(rand()) / RAND_MAX * spread - spread / 2.0f);
        float phi = central_phi + (static_cast<float>(rand()) / RAND_MAX * spread - spread / 2.0f);

        float x = sphere_radius * cos(phi) * cos(theta);
        float y = sphere_radius * cos(phi) * sin(theta);
        float z = sphere_radius * sin(phi);

        Eigen::Vector3d position(x, y, z);
        Eigen::Vector3d look_at(0.0, 0.0, 0.0);

        viewpoints.emplace_back(std::make_shared<visioncraft::Viewpoint>(position, look_at));
    }
    return viewpoints;
}

// Generate randomly distributed viewpoints on the sphere.
// Parameters:
// - num_viewpoints: Number of viewpoints to generate.
// - sphere_radius: Radius of the sphere on which viewpoints are placed.
// Returns: Vector of shared pointers to Viewpoint objects.
std::vector<std::shared_ptr<visioncraft::Viewpoint>> generateRandomViewpoints(int num_viewpoints, float sphere_radius) {
    std::vector<std::shared_ptr<visioncraft::Viewpoint>> viewpoints;

    for (int i = 0; i < num_viewpoints; ++i) {
        // Uniformly sample theta between 0 and 2*pi (azimuthal angle)
        float theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;

        // Uniformly sample phi between 0 and pi (polar angle)
        float phi = static_cast<float>(rand()) / RAND_MAX * M_PI;

        // Convert spherical coordinates (radius, theta, phi) to Cartesian coordinates
        float x = sphere_radius * sin(phi) * cos(theta);
        float y = sphere_radius * sin(phi) * sin(theta);
        float z = sphere_radius * cos(phi);

        Eigen::Vector3d position(x, y, z);
        Eigen::Vector3d look_at(0.0, 0.0, 0.0);

        viewpoints.emplace_back(std::make_shared<visioncraft::Viewpoint>(position, look_at));
    }
    return viewpoints;
}

// Compute the geodesic distance between two points on a sphere.
// Parameters:
// - point1, point2: Positions on the sphere.
// - sphere_radius: Radius of the sphere.
// Returns: Geodesic distance.
float computeGeodesicDistance(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, float sphere_radius) {
    // Normalize the points to unit vectors (this assumes points are given in global coordinates)
    Eigen::Vector3d p1_normalized = point1.normalized();
    Eigen::Vector3d p2_normalized = point2.normalized();

    // Compute the dot product
    float dotProduct = p1_normalized.dot(p2_normalized);

    // Clamp the dot product to avoid floating-point errors causing values outside [-1, 1]
    dotProduct = std::max(-1.0f, std::min(1.0f, dotProduct));

    // Compute the geodesic distance using the spherical law of cosines
    return sphere_radius * std::acos(dotProduct);
}

// Compute sigmoid function for weighting/normalization.
// Parameters:
// - x: Input value.
// - lambda: Steepness parameter.
// - shift: Shift parameter.
// Returns: Sigmoid output between 0 and 1.
float computeSigmoid(float x, float lambda = 20.0f, float shift = 0.75f)
 {
    float sigmoid_input = (x - shift) * lambda;
    return 1.0 / (1.0 + std::exp(-sigmoid_input));
}

// Put near the top of the file (globals allowed in your setup)
static bool  g_phase2 = false;     // set true once coverage==1 for ≥2 frames
static int   g_phase2_target = 2;  // prefer 2 views per voxel in phase 2
static float g_phase2_under  = 1.0f;  // pull weight if under-covered
static float g_phase2_over   = 1.2f;  // push weight if over-covered (slightly stronger)

// DISCRETE need pre-coverage; signed need post-coverage (same force formula)
// Attraction remains: sum( need / d * tangent )
// DROP-IN: replace your current smoothCoverageNeed with this
// DROP-IN: replace your current smoothCoverageNeed with this
// DROP-IN: replace your current smoothCoverageNeed(int hits)
inline float smoothCoverageNeed(int hits) {
    // Sharp logistic: need(0) ≈ 0.98, need(1) ≈ 0.02, need(2) ≈ ~0
    // k,t precomputed from n0=0.98, n1=0.02
    constexpr float k  = 7.78364086f;  // L(0.98) - L(0.02)
    constexpr float t  = 0.5f;         // L(0.98) / k
    const float   h    = float(std::max(0, hits));
    return 1.0f / (1.0f + std::exp(k * (h - t)));
}





inline double computeTotalNeed(
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap
) {
    double total = 0.0;
    for (const auto& kv : voxelToSphereMap) {
        const auto& key = kv.first;
        int hits = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        total += static_cast<double>(smoothCoverageNeed(hits)); // uses Change #2
    }
    return total; // larger = more uncovered/under-redundant mass
}

inline double mean(const std::deque<double>& q) {
    if (q.empty()) return 0.0;
    return std::accumulate(q.begin(), q.end(), 0.0) / static_cast<double>(q.size());
}

inline double slope_last_window(const std::deque<double>& q) {
    // simple end-to-start slope over the window (robust enough here)
    if (q.size() < 2) return 0.0;
    return (q.back() - q.front()) / static_cast<double>(q.size() - 1);
}


// Function to map voxels to sphere positions, handling occlusions.
// Parameters:
// - model: The visioncraft Model object.
// - sphere_radius: Radius of the surrounding sphere.
// - voxelToSphereMap: Output map from voxel keys to sphere positions.
// This function uses raycasting to detect self-occluded voxels and maps them to nearest unoccluded positions using KD-tree.
// void mapVoxelsToSphere(
//     visioncraft::Model& model,
//     float sphere_radius,
//     std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap
// ) {
//     const auto& voxelMap = model.getVoxelMap().getMap();
//     if (voxelMap.empty()) {
//         std::cerr << "[ERROR] No voxels available in the model." << std::endl;
//         return;
//     }
//     std::cout << "[INFO] Number of voxels in the model: " << voxelMap.size() << std::endl;

//     auto octree = model.getSurfaceShellOctomap();
//     if (!octree) {
//         std::cerr << "[ERROR] Octree is null. Cannot perform raycasting." << std::endl;
//         return;
//     }

//     double voxelSize = model.getVoxelSize(); // Assume this returns the edge length of the voxel cube
//     double diagonalLength = voxelSize * std::sqrt(3.0); // Diagonal length of the voxel cube

//     std::vector<Eigen::Vector3d> unoccludedPositions;
//     // Create a map of unoccluded positions to their keys
//     std::unordered_map<Eigen::Vector3d, octomap::OcTreeKey, Vector3dHash> unoccludedPositionsToKeys;
//     std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash> selfOccludedVoxels;

//     int numUnoccluded = 0;
//     int numSelfOccluded = 0;

//     // For each voxel
//     for (const auto& kv : voxelMap) {
//         const auto& voxel = kv.second;
//         const auto& key = kv.first;

//         Eigen::Vector3d voxelPosition = voxel.getPosition();
//         Eigen::Vector3d normal;
//         try {
//             normal = boost::get<Eigen::Vector3d>(model.getVoxelProperty(key, "normal"));
//         } catch (const boost::bad_get&) {
//             std::cerr << "[WARNING] Voxel at key " << key.k[0] << ", " << key.k[1] << ", " << key.k[2]
//                       << " does not have a normal property. Skipping." << std::endl;
//             continue;
//         }
//         normal.normalize(); // Ensure the normal is unit length

//         // Compute the intersection point of the ray from voxelPosition along normal with the sphere
//         double a = 1.0; // normal is normalized
//         double b = 2.0 * voxelPosition.dot(normal);
//         double c = voxelPosition.squaredNorm() - sphere_radius * sphere_radius;

//         double discriminant = b * b - 4.0 * a * c;
//         if (discriminant < 0) {
//             std::cerr << "[DEBUG] No intersection for voxel at " << voxelPosition.transpose() << std::endl;
//             continue;
//         }

//         double sqrt_disc = std::sqrt(discriminant);
//         double t1 = (-b + sqrt_disc) / (2.0 * a);
//         double t2 = (-b - sqrt_disc) / (2.0 * a);

//         double t = std::max(t1, t2); // Choose the larger t to ensure the ray goes outward
//         if (t <= 0) {
//             std::cerr << "[DEBUG] Intersection is behind the voxel at " << voxelPosition.transpose() << std::endl;
//             continue;
//         }

//         Eigen::Vector3d spherePoint = voxelPosition + t * normal;

//         // Adjust the starting point of the ray to avoid self-occlusion
//         Eigen::Vector3d rayStart = voxelPosition + diagonalLength * normal; // Start raycasting after the diagonal

//         bool occluded = false;
//         octomap::point3d origin(rayStart.x(), rayStart.y(), rayStart.z());
//         octomap::point3d direction(normal.x(), normal.y(), normal.z());
//         double maxRange = t - diagonalLength; // Reduce range to account for starting offset

//         octomap::point3d end;
//         bool hit = octree->castRay(origin, direction, end, true, maxRange);

//         if (hit) {
//             occluded = true;
//             numSelfOccluded++;
//         }

//         if (!occluded) {
//             voxelToSphereMap[key] = spherePoint;
//             unoccludedPositions.push_back(voxelPosition);
//             unoccludedPositionsToKeys[voxelPosition] = key;
//             numUnoccluded++;
//         } else {
//             selfOccludedVoxels[key] = spherePoint;
//         }
//     }

//     std::cout << "[INFO] Number of unoccluded voxels: " << numUnoccluded << std::endl;
//     std::cout << "[INFO] Number of self-occluded voxels: " << numSelfOccluded << std::endl;

//     // Use KD-tree for finding the nearest unoccluded neighbor for self-occluded voxels
//     if (!unoccludedPositions.empty()) {
//         auto kdtree = std::make_shared<open3d::geometry::KDTreeFlann>();
//         auto pointCloud = std::make_shared<open3d::geometry::PointCloud>();
//         pointCloud->points_ = unoccludedPositions;
//         kdtree->SetGeometry(*pointCloud);

//         for (const auto& kv : selfOccludedVoxels) {
//             const auto& key = kv.first;
//             Eigen::Vector3d occludedPosition = model.getVoxel(key)->getPosition();
//             Eigen::Vector3d occludedNormal = boost::get<Eigen::Vector3d>(model.getVoxelProperty(key, "normal"));
//             occludedNormal.normalize();

//             std::vector<int> indices;
//             std::vector<double> distances;

//             if (kdtree->SearchKNN(occludedPosition, 1, indices, distances) > 0) {
//                 int nearestIdx = indices[0];
//                 auto nearestVoxelPosition = unoccludedPositions[nearestIdx];
//                 auto nearestVoxelKey = unoccludedPositionsToKeys[nearestVoxelPosition];
//                 Eigen::Vector3d nearestNormal = boost::get<Eigen::Vector3d>(model.getVoxelProperty(nearestVoxelKey, "normal"));
//                 nearestNormal.normalize();

//                 bool successfullyMapped = false;
//                 Eigen::Vector3d spherePoint;

//                 for (double alpha = 0.1; alpha <= 1.0; alpha += 0.1) {
//                     Eigen::Vector3d modifiedNormal = (1.0 - alpha) * occludedNormal + alpha * nearestNormal;
//                     modifiedNormal.normalize();

//                     double a = 1.0;
//                     double b = 2.0 * occludedPosition.dot(modifiedNormal);
//                     double c = occludedPosition.squaredNorm() - sphere_radius * sphere_radius;

//                     double discriminant = b * b - 4.0 * a * c;
//                     if (discriminant < 0) continue;

//                     double sqrt_disc = std::sqrt(discriminant);
//                     double t1 = (-b + sqrt_disc) / (2.0 * a);
//                     double t2 = (-b - sqrt_disc) / (2.0 * a);

//                     double t = std::max(t1, t2);
//                     if (t <= 0) continue;

//                     spherePoint = occludedPosition + t * modifiedNormal;

//                     Eigen::Vector3d rayStart = occludedPosition + diagonalLength * modifiedNormal;
//                     octomap::point3d origin(rayStart.x(), rayStart.y(), rayStart.z());
//                     octomap::point3d direction(modifiedNormal.x(), modifiedNormal.y(), modifiedNormal.z());
//                     double maxRange = t - diagonalLength;

//                     octomap::point3d end;
//                     bool hit = octree->castRay(origin, direction, end, true, maxRange);

//                     if (!hit) {
//                         successfullyMapped = true;
//                         break;
//                     }
//                 }

//                 if (successfullyMapped) {
//                     voxelToSphereMap[key] = spherePoint;
//                 } else {
//                     // Fallback to old method
//                     voxelToSphereMap[key] = voxelToSphereMap[nearestVoxelKey];
//                 }
//             } else {
//                 std::cerr << "[WARNING] No nearest unoccluded neighbor found for self-occluded voxel at "
//                           << occludedPosition.transpose() << std::endl;
//             }
//         }

//         std::cout << "[INFO] Total voxels mapped to sphere: " << voxelToSphereMap.size() << std::endl;
//     } else {
//         std::cerr << "[ERROR] No unoccluded positions available to map self-occluded voxels." << std::endl;
//     }
// }


void mapVoxelsToSphere(
    visioncraft::Model& model,
    float sphere_radius,
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap
) {
    const auto& voxelMap = model.getVoxelMap().getMap();
    if (voxelMap.empty()) {
        std::cerr << "[ERROR] No voxels available in the model." << std::endl;
        return;
    }
    std::cout << "[INFO] Number of voxels in the model: " << voxelMap.size() << std::endl;

    auto octree = model.getSurfaceShellOctomap();
    if (!octree) {
        std::cerr << "[ERROR] Octree is null. Cannot perform raycasting." << std::endl;
        return;
    }

    voxelToSphereMap.clear();

    const double voxelSize      = model.getVoxelSize();           // edge length
    const double diagonalLength = voxelSize * std::sqrt(3.0);     // cube diagonal
    const double epsNorm        = 1e-12;

    // ---- tunables for pole flare ----
    const double pole_thresh         = 0.70;   // start flaring when |n·z| > this
    const double flare_gain          = 0.6;   // max blend toward lateral (0..1)
    const double flare_gamma         = 2.0;    // steeper curve => more flare near poles
    const double pole_min_theta_deg  = 15.0;   // skip θ=0 for strong poles (start at 15°)

    // Helpers (local lambdas)
    auto deg2rad = [](double d){ return d * M_PI / 180.0; };

    auto orthonormalBasis = [](const Eigen::Vector3d& n, Eigen::Vector3d& u, Eigen::Vector3d& v) {
        if (std::fabs(n.z()) < 0.9) u = n.cross(Eigen::Vector3d::UnitZ()).normalized();
        else                        u = n.cross(Eigen::Vector3d::UnitY()).normalized();
        v = n.cross(u).normalized();
    };

    auto hemisphereDir = [&](const Eigen::Vector3d& axis_unit, double theta, double phi) -> Eigen::Vector3d {
        // d = cosθ * axis + sinθ * (cosφ * u + sinφ * v)
        Eigen::Vector3d u, v;
        orthonormalBasis(axis_unit, u, v);
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        return (c * axis_unit + s * (std::cos(phi) * u + std::sin(phi) * v)).normalized();
    };

    auto sphereIntersectFar = [&](const Eigen::Vector3d& p, const Eigen::Vector3d& d, double R, double& t_out) -> bool {
        // d assumed unit, sphere centered at origin
        const double b = 2.0 * p.dot(d);
        const double c = p.squaredNorm() - R * R;
        const double disc = b*b - 4.0*c;
        if (disc < 0.0) return false;
        const double sqrt_disc = std::sqrt(disc);
        const double t1 = (-b - sqrt_disc) * 0.5;
        const double t2 = (-b + sqrt_disc) * 0.5;
        const double t_far = std::max(t1, t2);
        if (t_far <= 0.0) return false;
        t_out = t_far;
        return true;
    };

    auto rayHitsBeforeSphere = [&](const Eigen::Vector3d& start, const Eigen::Vector3d& dir_unit, double maxRange) -> bool {
        octomap::point3d origin((float)start.x(), (float)start.y(), (float)start.z());
        octomap::point3d direction((float)dir_unit.x(), (float)dir_unit.y(), (float)dir_unit.z());
        octomap::point3d end;
        const bool ignoreUnknownCells = true; // keep your original permissive setting
        return octree->castRay(origin, direction, end, ignoreUnknownCells, (float)maxRange);
    };

    // --- ANGLE SETS (15° θ, 15° φ) ---
    // base θ grid
    std::vector<double> thetas_base; thetas_base.reserve(7); // 0,15,30,45,60,75,90
    for (int d = 0; d <= 90; d += 15) thetas_base.push_back(deg2rad((double)d));

    // φ: 0..345 in 15° steps, later shifted per-voxel by φ0
    std::vector<double> phis0; phis0.reserve(24);
    for (int d = 0; d < 360; d += 15) phis0.push_back(deg2rad((double)d));
    std::vector<double> phis_order; phis_order.reserve(phis0.size());
    {
        const int N = (int)phis0.size(); // 24
        phis_order.push_back(phis0[0]); // 0°
        for (int k = 1; k <= N/2; ++k) {
            int i_pos = ( k) % N;   // +15, +30, ...
            int i_neg = (N-k) % N;  // -15, -30, ...
            if ((int)phis_order.size() < N) phis_order.push_back(phis0[i_pos]);
            if ((int)phis_order.size() < N && i_neg != 0) phis_order.push_back(phis0[i_neg]);
        }
        for (int i = 0; i < N && (int)phis_order.size() < N; ++i)
            phis_order.push_back(phis0[i]);
    }

    // Stats
    int numMapped = 0, numSkippedNoNormal = 0, numFailedAll = 0;

    for (const auto& kv : voxelMap) {
        const auto& key   = kv.first;
        const auto& voxel = kv.second;

        const Eigen::Vector3d p = voxel.getPosition();

        Eigen::Vector3d n;
        try {
            n = boost::get<Eigen::Vector3d>(model.getVoxelProperty(key, "normal"));
        } catch (const boost::bad_get&) {
            ++numSkippedNoNormal;
            continue;
        }
        if (!std::isfinite(n.x()) || !std::isfinite(n.y()) || !std::isfinite(n.z()) || n.norm() < epsNorm) {
            ++numSkippedNoNormal;
            continue;
        }
        n.normalize();

        // ---------- Pole flare (tiny tilt away from ±Z, toward lateral / equator) ----------
        const double pole = std::abs(n.z());                         // 0..1
        const double w    = flare_gain * std::pow(pole, flare_gamma); // 0..flare_gain
        Eigen::Vector3d lateral(p.x(), p.y(), 0.0);                  // use voxel's XY azimuth
        if (lateral.squaredNorm() < 1e-16) lateral = Eigen::Vector3d(n.x(), n.y(), 0.0);
        if (lateral.squaredNorm() > 0.0) lateral.normalize();

        // Blend: axis = (1-w)*n + w*lateral  (keeps hemisphere tied to the normal but nudges off the pole)
        Eigen::Vector3d axis = ((1.0 - w) * n + w * lateral).normalized();

        // Per-voxel φ shift so φ=0 aligns with lateral in the (axis) frame; accelerates “flare” preference
        Eigen::Vector3d u_axis, v_axis; orthonormalBasis(axis, u_axis, v_axis);
        double phi0 = 0.0;
        if (lateral.squaredNorm() > 0.0) {
            const double cx = lateral.dot(u_axis);
            const double cy = lateral.dot(v_axis);
            phi0 = std::atan2(cy, cx); // rotate φ so that search starts along lateral azimuth
        }

        // Per-voxel θ schedule (skip θ=0 for strong poles to avoid exact-pole pileups)
        std::vector<double> thetas = thetas_base;
        if (pole > pole_thresh) {
            const double tmin = deg2rad(pole_min_theta_deg);
            // Move θ=0 behind θ=15°, or simply replace the first entry with tmin
            thetas[0] = tmin;
        }

        bool mapped = false;

        for (double theta : thetas) {
            if (theta == 0.0) {
                // Try exactly along the flared axis first (phi is irrelevant)
                Eigen::Vector3d dir = axis;
                double t = 0.0;
                if (sphereIntersectFar(p, dir, sphere_radius, t)) {
                    const double maxRange = std::max(0.0, t - diagonalLength);
                    const Eigen::Vector3d start = p + diagonalLength * dir;
                    const bool hit = rayHitsBeforeSphere(start, dir, maxRange);
                    if (!hit) {
                        voxelToSphereMap[key] = p + t * dir;
                        ++numMapped;
                        mapped = true;
                    }
                }
                if (mapped) break;
                continue; // fall through to next theta if blocked
            }

            // theta > 0: sweep φ around the flared axis, starting at the lateral azimuth (phi0)
            for (double phi : phis_order) {
                const Eigen::Vector3d dir = hemisphereDir(axis, theta, phi0 + phi);
                double t = 0.0;
                if (!sphereIntersectFar(p, dir, sphere_radius, t)) continue;

                const double maxRange = std::max(0.0, t - diagonalLength);
                const Eigen::Vector3d start = p + diagonalLength * dir;
                const bool hit = rayHitsBeforeSphere(start, dir, maxRange);
                if (!hit) {
                    voxelToSphereMap[key] = p + t * dir;
                    ++numMapped;
                    mapped = true;
                    break;
                }
            }
            if (mapped) break;
        }

        if (!mapped) {
            ++numFailedAll; // remains unmapped; keep behavior
            // Optional fallback:
            /*
            double t_fallback = 0.0;
            if (sphereIntersectFar(p, axis, sphere_radius, t_fallback)) {
                voxelToSphereMap[key] = p + t_fallback * axis;
                ++numMapped;
                --numFailedAll;
            }
            */
        }
    }

    std::cout << "[INFO] Mapped with pole-flare: " << numMapped
              << " | No-normal: " << numSkippedNoNormal
              << " | Failed after sweep: " << numFailedAll
              << " | Total saved: " << voxelToSphereMap.size()
              << " | flare_gain=" << 0.55
              << " | pole_min_theta=" << pole_min_theta_deg << " deg"
              << std::endl;
}

/**
 * FIELD POTENTIAL AND DYNAMICS FUNCTIONS
 */

// Reset potential values for all voxels to 0.
// Parameters:
// - model: The visioncraft Model object.
void resetPotentials(visioncraft::Model& model) {
    const auto& voxelMap = model.getVoxelMap().getMap();
    for (const auto& kv : voxelMap) {
        const auto& key = kv.first;
        model.setVoxelProperty(key, "potential", 0.0f);
    }
}

// Compute voxel potentials based on visibility and geodesic distances.
// Adapted mathematical model: potential =  (1 - sigmoid(visibility)) * sum(log(d)) over viewpoints.
// Parameters:
// - model: The visioncraft Model object.
// - voxelToSphereMap: Map from voxel keys to sphere positions.
// - viewpoints: List of viewpoints.
// - sphere_radius: Sphere radius.
// Compute voxel potentials based on *live* hit counts (same need as forces).
// potential(voxel) = smoothCoverageNeed(hits) * sum_viewpoints log(geodesic_distance + eps)
void computeVoxelPotentials(
    visioncraft::Model& model,
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    float sphere_radius,
    int sample_rate,
    const std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>& hit_count
) {
    const float epsilon = 1e-6f;
    int count = 0;

    for (const auto& kv : voxelToSphereMap) {
        if (++count % sample_rate != 0) continue;

        const auto& key         = kv.first;
        const Eigen::Vector3d& s_pos = kv.second;

        // Use *live* hits from this iteration
        int hits = 0;
        auto it = hit_count.find(key);
        if (it != hit_count.end()) hits = it->second;

        const float need = smoothCoverageNeed(hits);
        if (need <= 0.0f) { model.setVoxelProperty(key, "potential", 0.0f); continue; }

        float pot_sum = 0.0f;
        for (const auto& vp : viewpoints) {
            const float gd = computeGeodesicDistance(vp->getPosition(), s_pos, sphere_radius);
            pot_sum += std::log(gd + epsilon);
        }

        model.setVoxelProperty(key, "potential", need * pot_sum);
    }
}


// Compute attractive force for a viewpoint based on voxel potentials.
// Adapted mathematical model: force = sum [  (1 - sigmoid(visibility)) / d * dir ] over voxels.
// Parameters:
// - model: The visioncraft Model object.
// - voxelToSphereMap: Map from voxel keys to sphere positions.
// - viewpoint: The viewpoint to compute force for.
// - sphere_radius: Sphere radius.
// Returns: Attractive force vector.
Eigen::Vector3d computeAttractiveForce(
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>&              hit_count, // NEW
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    float sphere_radius,
    int sample_rate
) {
    const float epsilon = 1e-6f;
    Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
    const Eigen::Vector3d vp = viewpoint->getPosition();
    int count = 0;

    for (const auto& kv : voxelToSphereMap) {
        if (++count % sample_rate != 0) continue;

        const auto& key   = kv.first;
        const auto& s_pos = kv.second;

        // Use the *current iteration* hits for smoothCoverageNeed
        int hits = 0;
        auto it = hit_count.find(key);
        if (it != hit_count.end()) hits = it->second;

        const float need = smoothCoverageNeed(hits);
        if (need <= 0.0f) continue;

        const float geod = computeGeodesicDistance(vp, s_pos, sphere_radius);
        if (geod < 1e-8f) continue;

        // Tangent direction on the sphere (from vp toward s_pos)
        Eigen::Vector3d d = s_pos - vp;
        Eigen::Vector3d n = vp.normalized();
        Eigen::Vector3d tangent = d - (d.dot(n)) * n;
        const double norm_t = tangent.norm();
        if (norm_t <= 1e-12) continue;
        tangent /= norm_t;

        // Logarithmic gradient model: weight ~ need / d
        const float weight = need / (geod + epsilon);
        total_force += weight * tangent;
    }

    // IMPORTANT: return the raw sum (no normalization), keeps 2D behavior
    return total_force;
}


// Compute repulsive force between viewpoints.
// Adapted mathematical model: force = sum [ amplitude * d / sigma^2 * exp(-d^2 / 2 sigma^2) * -dir ] over other viewpoints.
// Parameters:
// - viewpoints: List of all viewpoints.
// - current_viewpoint: The viewpoint to compute repulsion for.
// - sphere_radius: Sphere radius.
// - sigma: Gaussian spread for repulsion.
// Returns: Repulsive force vector.
// Repulsion = Gaussian value, max at d=0, smooth decay with d.
// Good if you want very strong short-range push and gentle long-range effect.
Eigen::Vector3d computeRepulsiveForce(
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    const std::shared_ptr<visioncraft::Viewpoint>& current_viewpoint,
    float sphere_radius,
    float sigma
) {
    const float epsilon = 1e-6f;
    const float amp = 1.0f; // tune with k_rep outside
    Eigen::Vector3d total = Eigen::Vector3d::Zero();

    const Eigen::Vector3d p = current_viewpoint->getPosition();
    const Eigen::Vector3d n = p.normalized();

    for (const auto& other : viewpoints) {
        if (other.get() == current_viewpoint.get()) continue;

        const Eigen::Vector3d q = other->getPosition();
        const float d = computeGeodesicDistance(p, q, sphere_radius);
        if (d < epsilon) continue;

        // Tangent direction (from current toward other)
        Eigen::Vector3d t = (q - p) - ((q - p).dot(n)) * n;
        const double tn = t.norm(); if (tn < 1e-12) continue;
        t /= tn;

        const float g = std::exp(-(d*d) / (2.0f * sigma * sigma)); // max at d=0
        total -= (amp * g) * t; // repel away from the other
    }

    // Optional averaging: comment out if you want "crowding stiffness" to scale with N
    int denom = std::max(1, int(viewpoints.size()) - 1);
    total /= float(denom);

    return total;
}



/**
 * SIMULATION FUNCTIONS
 */

// Update viewpoint position and optionally orientation.
// Parameters:
// - viewpoint: Viewpoint to update.
// - new_position: New position vector.
// - sphere_radius: Sphere radius for normalization.
// - set_orientation: If true, reset orientation to look at center.
void updateViewpointState(
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    const Eigen::Vector3d& new_position,
    float sphere_radius,
    bool set_orientation = false) 
{
    Eigen::Vector3d normalized_position = sphere_radius * new_position.normalized();
    viewpoint->setPosition(normalized_position);
    if (set_orientation){
        viewpoint->setLookAt(Eigen::Vector3d(0.0, 0.0, 0.0), -Eigen::Vector3d::UnitZ());
    }
}

// Add a new viewpoint to the simulation.
// Parameters:
// - viewpoints: List to add to.
// - visibilityManager: Visibility manager.
// - visualizer: Visualizer object.
// - position: Initial position.
// - look_at: Look-at point.
// - sphere_radius: Sphere radius.
void addNewViewpoint(
    std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    std::shared_ptr<visioncraft::VisibilityManager> visibilityManager,
    visioncraft::Visualizer& visualizer,
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& look_at,
    float sphere_radius) 
{
    // Create a new viewpoint
    auto new_viewpoint = std::make_shared<visioncraft::Viewpoint>(position, look_at);
    new_viewpoint->setDownsampleFactor(8.0);
    new_viewpoint->setFarPlane(900);
    new_viewpoint->setNearPlane(300);
    
    // Normalize position to ensure it's on the sphere's surface
    Eigen::Vector3d normalized_position = sphere_radius * position.normalized();
    new_viewpoint->setPosition(normalized_position);
    
    // Add to the viewpoints list
    viewpoints.push_back(new_viewpoint);

    // Add to the visibility manager
    visibilityManager->trackViewpoint(new_viewpoint);

    // Add the new viewpoint to the visualizer
    visualizer.addViewpoint(*new_viewpoint, false, true);
}

// Interpolate potentials on a sphere for visualization.
// Parameters:
// - model: The visioncraft Model object.
// - voxelToSphereMap: Voxel to sphere map.
// - property_name: Property to interpolate (e.g., "potential").
// - sphere_radius: Sphere radius.
// Returns: VTK PolyData with interpolated potentials.
vtkSmartPointer<vtkPolyData> computeInterpolatedPotentialsOnSphere(
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::string& property_name,
    float sphere_radius)
{
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetRadius(sphere_radius);
    sphereSource->SetThetaResolution(100);
    sphereSource->SetPhiResolution(100);
    sphereSource->Update();

    vtkSmartPointer<vtkPolyData> spherePolyData = sphereSource->GetOutput();
    vtkSmartPointer<vtkPoints> sphereVertices = spherePolyData->GetPoints();

    if (!sphereVertices) {
        throw std::runtime_error("Failed to generate sphere vertices.");
    }

    // Prepare KD-tree and mapping
    std::vector<Eigen::Vector3d> mappedPositions;
    std::unordered_map<int, octomap::OcTreeKey> spherePointToKeyMap;

    int pointIndex = 0;
    for (const auto& kv : voxelToSphereMap) {
        mappedPositions.push_back(kv.second);
        spherePointToKeyMap[pointIndex++] = kv.first;
    }

    auto kdtree = std::make_shared<open3d::geometry::KDTreeFlann>();
    auto pointCloud = std::make_shared<open3d::geometry::PointCloud>();
    pointCloud->points_ = mappedPositions;
    kdtree->SetGeometry(*pointCloud);

    vtkSmartPointer<vtkFloatArray> interpolatedPotentials = vtkSmartPointer<vtkFloatArray>::New();
    interpolatedPotentials->SetName("potential");

    float maxPotential = std::numeric_limits<float>::lowest();

    for (vtkIdType i = 0; i < sphereVertices->GetNumberOfPoints(); ++i) {
        double sphereVertex[3];
        sphereVertices->GetPoint(i, sphereVertex);

        Eigen::Vector3d vertexPosition(sphereVertex[0], sphereVertex[1], sphereVertex[2]);
        std::vector<int> indices;
        std::vector<double> distances;

        // Query KD-tree
        int numFound = kdtree->SearchKNN(vertexPosition, 5, indices, distances);

        float potentialSum = 0.0f;
        double weightSum = 0.0;

        for (size_t j = 0; j < indices.size(); ++j) {
            double weight = 1.0 / (std::sqrt(distances[j]) + 1e-6);
            const auto& key = spherePointToKeyMap[indices[j]];
            float potential = 0.0f;

            try {
                potential = boost::get<float>(model.getVoxelProperty(key, property_name));
            } catch (const boost::bad_get&) {
                potential = 0.0f;
            }
           potentialSum += weight * potential;
            weightSum += weight;
        }

        float interpolatedPotential = (weightSum > 0) ? potentialSum / weightSum : 0.0f;
        interpolatedPotentials->InsertNextValue(interpolatedPotential);
        maxPotential = std::max(maxPotential, interpolatedPotential);
    }
    std::cout << "Max Potential: " << maxPotential << std::endl;

    spherePolyData->GetPointData()->AddArray(interpolatedPotentials);
    return spherePolyData;
}

Eigen::Vector3d computeStep(
    const Eigen::Vector3d& force,
    Eigen::Vector3d& m,
    Eigen::Vector3d& v,
    int timestep,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    bool use_adam
) {
    if (!use_adam) {
        // ---- Plain SGD with trust-region step length ----
        // Keep the force direction, but ensure a minimum step length for responsiveness.
        // Also cap the maximum to avoid wild jumps when forces spike.
        const float MIN_STEP = 6.0f;   // tweak: 3–10 gives snappy but controlled motion
        const float MAX_STEP = 20.0f;  // safety cap
        const float eps_norm = 1e-12f;

        float f = force.norm();
        if (f < eps_norm) return Eigen::Vector3d::Zero();

        // Base proportional step
        float step_len = lr * f;

        // Trust-region clamp
        if (step_len < MIN_STEP) step_len = MIN_STEP;
        if (step_len > MAX_STEP) step_len = MAX_STEP;

        return (step_len / f) * force; // same direction as force
    }

    // ---- ADAM update (momentum + RMS scaling) ----
    m = beta1 * m + (1.0f - beta1) * force;
    v = beta2 * v + (1.0f - beta2) * force.cwiseProduct(force);

    float beta1_pow = std::pow(beta1, timestep);
    float beta2_pow = std::pow(beta2, timestep);
    Eigen::Vector3d m_hat = m / (1.0f - beta1_pow);
    Eigen::Vector3d v_hat = v / (1.0f - beta2_pow);

    return lr * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + epsilon * Eigen::Vector3d::Ones());
}


// C++14-safe path -> stem ("../models/gorilla.ply" -> "gorilla")
static inline std::string file_stem(const std::string& path) {
    // find last path separator
    size_t slash = path.find_last_of("/\\");
    const size_t start = (slash == std::string::npos) ? 0 : slash + 1;

    // find last dot *after* the last separator
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || dot < start) dot = path.size();

    return path.substr(start, dot - start);
}


// ===== REPLACE your entire exportFinalMetrics(...) with this NO-SAMPLING version =====
void exportFinalMetrics(
    const visioncraft::Model& model,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    const std::string& output_dir,
    int /*seed not used anymore*/
) {
    // ---- 0) Gather ALL voxels (no sampling)
    std::vector<octomap::OcTreeKey> all_voxels;
    all_voxels.reserve(model.getVoxelMap().getMap().size());
    for (const auto& pair : model.getVoxelMap().getMap()) {
        all_voxels.push_back(pair.first);
    }
    const size_t V = all_voxels.size();
    const int     N = static_cast<int>(viewpoints.size());

    // ---- 1) Build exact visibility sets per viewpoint from hit results
    std::vector<std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>> vis_sets(N);
    for (int j = 0; j < N; ++j) {
        const auto& hits = viewpoints[j]->getHitResults();
        for (const auto& hit : hits) {
            if (hit.second) vis_sets[j].insert(hit.first);
        }
    }

    // ---- 2) Build full assignments matrix (V × N) and stream-write to CSV
    // NOTE: This can be huge. Keep only if you truly need the full matrix on disk.
    {
        std::ofstream assign_csv(output_dir + "/viewpoint_point_assignments.csv");
        for (size_t s = 0; s < V; ++s) {
            const auto& key = all_voxels[s];
            for (int j = 0; j < N; ++j) {
                const bool seen = (vis_sets[j].find(key) != vis_sets[j].end());
                assign_csv << (seen ? '1' : '0');
                if (j < N - 1) assign_csv << ",";
            }
            assign_csv << "\n";
        }
    }

    // ---- 3) viewpoint_contribution_hist: #voxels seen by each viewpoint (over ALL voxels)
    {
        std::ofstream contrib_csv(output_dir + "/viewpoint_contribution_hist.csv");
        for (int j = 0; j < N; ++j) {
            contrib_csv << vis_sets[j].size() << "\n";
        }
    }

    // ---- 4) point_redundancy_hist over ALL voxels (include zeros)
    // Build a voxel -> count map once, then produce histogram.
    std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> voxel_k;
    voxel_k.reserve(V * 1.1);
    for (int j = 0; j < N; ++j) {
        for (const auto& key : vis_sets[j]) {
            ++voxel_k[key];
        }
    }
    int k_max = 0;
    for (size_t s = 0; s < V; ++s) {
        int k = 0;
        auto it = voxel_k.find(all_voxels[s]);
        if (it != voxel_k.end()) k = it->second;
        if (k > k_max) k_max = k;
    }
    std::vector<int> point_red_hist(k_max + 1, 0);
    for (size_t s = 0; s < V; ++s) {
        int k = 0;
        auto it = voxel_k.find(all_voxels[s]);
        if (it != voxel_k.end()) k = it->second;
        ++point_red_hist[k];
    }
    {
        std::ofstream red_hist_csv(output_dir + "/point_redundancy_hist.csv");
        for (int c : point_red_hist) red_hist_csv << c << "\n";
    }

    // ---- 5) viewpoint_overlap_matrix: exact Jaccard over ALL voxels using set ops
    {
        std::ofstream overlap_csv(output_dir + "/viewpoint_overlap_matrix.csv");
        const float eps = 1e-6f;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    overlap_csv << "1.0";
                } else {
                    // iterate over smaller set for intersection
                    const auto& A = vis_sets[i];
                    const auto& B = vis_sets[j];
                    const auto* S = (A.size() < B.size()) ? &A : &B;
                    const auto* L = (S == &A) ? &B : &A;

                    int inter = 0;
                    for (const auto& key : *S) {
                        if (L->find(key) != L->end()) ++inter;
                    }
                    const int uni = int(A.size() + B.size() - inter);
                    const float jac = float(inter) / (float(uni) + eps);
                    overlap_csv << jac;
                }
                if (j < N - 1) overlap_csv << ",";
            }
            overlap_csv << "\n";
        }
    }

    // ---- 6) functional_isolation_flags: avg row overlap < 5%
    {
        std::ofstream isolation_csv(output_dir + "/functional_isolation_flags.csv");
        const float eps = 1e-6f;
        for (int i = 0; i < N; ++i) {
            float sum_ov = 0.0f;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                // Jaccard(i,j)
                const auto& A = vis_sets[i];
                const auto& B = vis_sets[j];
                const auto* S = (A.size() < B.size()) ? &A : &B;
                const auto* L = (S == &A) ? &B : &A;

                int inter = 0;
                for (const auto& key : *S) {
                    if (L->find(key) != L->end()) ++inter;
                }
                const int uni = int(A.size() + B.size() - inter);
                sum_ov += float(inter) / (float(uni) + eps);
            }
            const float avg_ov = (N > 1) ? (sum_ov / float(N - 1)) : 0.0f;
            const bool isolated = (avg_ov < 0.05f);
            isolation_csv << (isolated ? "1" : "0") << "\n";
        }
    }

    // ---- 7) final_viewpoints (positions) and potential_field_viewpoints (pose)
    {
        std::ofstream final_pos_csv(output_dir + "/final_viewpoints.csv");
        final_pos_csv << "x,y,z\n";
        for (const auto& vp : viewpoints) {
            auto p = vp->getPosition();
            final_pos_csv << p.x() << "," << p.y() << "," << p.z() << "\n";
        }
    }
    {
        std::ofstream final_vp_csv(output_dir + "/potential_field_viewpoints.csv");
        final_vp_csv << "x,y,z,qx,qy,qz,qw\n";
        for (const auto& vp : viewpoints) {
            auto p = vp->getPosition();
            auto q = vp->getOrientationQuaternion();
            final_vp_csv << p.x() << "," << p.y() << "," << p.z() << ","
                         << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        }
    }
}


bool is_equilibrium(const std::deque<float>& recent_moves, const std::deque<float>& recent_potentials, int window_size, float initial_total_potential) {
    if (recent_moves.size() < static_cast<size_t>(window_size) || recent_potentials.size() < static_cast<size_t>(window_size)) return false;

    float avg_move = std::accumulate(recent_moves.begin(), recent_moves.end(), 0.0f) / window_size;
    float pot_rate = (recent_potentials.front() - recent_potentials.back()) / (window_size - 1.0f);

    float move_threshold = 0.5f;
    float pot_rate_epsilon = 0.01f * initial_total_potential;

    return avg_move < move_threshold && std::abs(pot_rate) < pot_rate_epsilon;
}

struct VisStats {
    size_t all_voxels = 0;
    size_t num_covered = 0;
    size_t total_hits = 0;
    size_t num_redundant = 0;
    double coverage = 0.0;
    double redundancy = 0.0;
    double affinity = 0.0;
    std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> hit_count;
    std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>      covered;
};

struct PFWindows {
    int    window_size = 12;
    float  T_motion_rough = 2.5f;
    float  T_motion_fine  = 0.8f;
    double initial_total_need = 1.0;
    std::deque<float> recent_moves;
    std::deque<float> recent_total_need;
};

inline VisStats computeVisibilityStats(
    const visioncraft::Model& model,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints
) {
    VisStats s;
    s.all_voxels = model.getVoxelMap().getMap().size();

    for (const auto& vp : viewpoints) {
        const auto& hits = vp->getHitResults();
        for (const auto& h : hits) {
            if (h.second) {
                s.hit_count[h.first]++;
                s.covered.insert(h.first);
            }
        }
    }
    s.num_covered = s.covered.size();
    for (const auto& kv : s.hit_count) {
        s.total_hits += kv.second;
        if (kv.second > 1) ++s.num_redundant;
    }

    s.coverage   = s.all_voxels ? double(s.num_covered) / s.all_voxels : 0.0;
    s.redundancy = s.all_voxels ? double(s.num_redundant) / s.all_voxels : 0.0;
    s.affinity   = s.num_covered ? double(s.total_hits) / s.num_covered : 0.0;
    return s;
}

inline double computeTotalNeedFromHits(
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>&              hit_count
) {
    double total_need = 0.0;
    for (const auto& kv : voxelToSphereMap) {
        auto it = hit_count.find(kv.first);
        int hits = (it == hit_count.end()) ? 0 : it->second;
        total_need += static_cast<double>(smoothCoverageNeed(hits));
    }
    return total_need;
}

inline octomap::OcTreeKey argmaxPotentialKey(const visioncraft::Model& model) {
    float best = std::numeric_limits<float>::lowest();
    octomap::OcTreeKey best_key;
    for (const auto& kv : model.getVoxelMap().getMap()) {
        float pot = boost::get<float>(model.getVoxelProperty(kv.first, "potential"));
        if (pot > best) { best = pot; best_key = kv.first; }
    }
    return best_key;
}

inline void updateMotionWindow(PFWindows& w, float avg_move) {
    w.recent_moves.push_back(avg_move);
    if (w.recent_moves.size() > static_cast<size_t>(w.window_size)) w.recent_moves.pop_front();
}

inline void updateNeedWindow(PFWindows& w, float total_need) {
    w.recent_total_need.push_back(total_need);
    if (w.recent_total_need.size() > static_cast<size_t>(w.window_size)) w.recent_total_need.pop_front();
}

inline float normalizedNeedSlope(const PFWindows& w) {
    if (w.recent_total_need.size() < 2) return 1000.0f;
    float start = w.recent_total_need.front();
    float end   = w.recent_total_need.back();
    float slope = (end - start) / std::max(1.0f, float(w.window_size - 1));
    return float(slope / std::max(1e-9, w.initial_total_need));
}

inline bool shouldInsertNBV(
    const PFWindows& w, double coverage_like,
    float need_rate_norm, bool smart, int iter, int insert_every,
    int iters_since_insert, int max_stage_iterations
) {
    if (!smart) return (iter % insert_every == 0);

    bool window_ready = (w.recent_moves.size() == size_t(w.window_size)) &&
                        (w.recent_total_need.size() == size_t(w.window_size));

    float T_motion = (coverage_like >= 0.95) ? w.T_motion_fine : w.T_motion_rough;
    float avg_move = 0.0f;
    if (!w.recent_moves.empty()) {
        avg_move = std::accumulate(w.recent_moves.begin(), w.recent_moves.end(), 0.0f) / float(w.recent_moves.size());
    }
    bool motion_ok    = window_ready && (avg_move < T_motion);
    bool potential_ok = window_ready && (std::abs(need_rate_norm) < 0.002f); // knob

    // Faster variant: OR in early/mid stages; AND near the end
    bool either_ok = motion_ok || potential_ok;
    bool main_gate = (coverage_like < 0.93) ? either_ok : (motion_ok && potential_ok);

    // Time-since-last-insert backstop
    if (!main_gate && iters_since_insert > max_stage_iterations) return true;

    return main_gate;
}

inline int decideSampleRate(double coverage) {
    if (coverage < 0.85) return 3;
    if (coverage < 0.93) return 2;
    return 1;
}

inline float coverageLike(const PFWindows& w) {
    if (w.recent_total_need.empty()) return 0.0f;
    float last_need = w.recent_total_need.back();
    return 1.0f - std::min(1.0f, last_need / float(std::max(1e-9, w.initial_total_need)));
}

// --- NBV by Potential Mass ---------------------------------------------------
struct NBVMassResult {
    Eigen::Vector3d pos;                                   // insertion position on sphere
    octomap::OcTreeKey rep_key;                            // representative key in the cluster
    double mass = 0.0;                                     // summed potential in window
    double radius_deg = 0.0;                               // window radius used (deg)
};

// Pick NBV as the densest high-potential cluster (highest potential *mass*).
// topK: how many highest-potential samples to consider
// ang_radius_deg: angular radius of the local window on the sphere (degrees)
NBVMassResult pickNBVByPotentialMass(
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    float sphere_radius,
    int topK,
    double ang_radius_deg
) {
    NBVMassResult out; out.radius_deg = ang_radius_deg;
    if (voxelToSphereMap.empty()) return out;

    struct Item { octomap::OcTreeKey key; float pot; Eigen::Vector3d dir; };
    std::vector<Item> items; items.reserve(voxelToSphereMap.size());

    // collect potentials + unit directions
    for (const auto& kv : voxelToSphereMap) {
        float pot = 0.0f;
        try { pot = boost::get<float>(model.getVoxelProperty(kv.first, "potential")); }
        catch (...) { pot = 0.0f; }
        if (!std::isfinite(pot)) pot = 0.0f;

        Eigen::Vector3d dir = kv.second.normalized();
        if (!dir.allFinite()) continue;

        items.push_back({ kv.first, pot, dir });
    }
    if (items.empty()) return out;

    // keep only top-K by potential (partial sort)
    const int K = std::min<int>(topK, (int)items.size());
    std::nth_element(items.begin(), items.begin()+K, items.end(),
                     [](const Item& a, const Item& b){ return a.pot > b.pot; });
    items.resize(K);

    // scan neighborhood mass for each candidate (O(K^2))
    const double ang_rad = ang_radius_deg * M_PI / 180.0;
    const double cos_thr = std::cos(ang_rad);

    double best_mass = -1.0;
    size_t best_i = 0;
    Eigen::Vector3d best_dir = Eigen::Vector3d::UnitX();

    for (size_t i = 0; i < items.size(); ++i) {
        const Eigen::Vector3d& di = items[i].dir;
        double mass = 0.0;
        Eigen::Vector3d wdir = Eigen::Vector3d::Zero();

        for (size_t j = 0; j < items.size(); ++j) {
            const Eigen::Vector3d& dj = items[j].dir;
            // geodesic window: angle(di,dj) <= ang_rad  <=>  di·dj >= cos_thr
            if (di.dot(dj) >= cos_thr) {
                const double w = (double)items[j].pot;
                mass += w;
                wdir += w * dj;
            }
        }

        if (mass > best_mass && wdir.allFinite() && wdir.norm() > 0.0) {
            best_mass = mass;
            best_i = i;
            best_dir = wdir.normalized();
        }
    }

    out.mass    = std::max(0.0, best_mass);
    out.rep_key = items[best_i].key;
    out.pos     = sphere_radius * best_dir;
    return out;
}


// ======= C++14-safe helpers + scoring + local refinement =======
#include <Eigen/Dense>

using Vec3 = Eigen::Vector3d;

static inline double deg2rad(double d){ return d * M_PI / 180.0; }

static inline void tangentBasis(const Vec3& n, Vec3& t1, Vec3& t2){
    if (std::fabs(n.z()) < 0.9) t1 = n.cross(Vec3::UnitZ()).normalized();
    else                        t1 = n.cross(Vec3::UnitY()).normalized();
    t2 = n.cross(t1).normalized();
}

// Move unit vector n by a small geodesic offset (radius_deg) along azimuth phi
static inline Vec3 offsetOnSphere(const Vec3& n, double radius_deg, double phi){
    Vec3 t1, t2; tangentBasis(n, t1, t2);
    const double th = deg2rad(radius_deg);
    const double c  = std::cos(th), s = std::sin(th);
    Vec3 lateral = std::cos(phi)*t1 + std::sin(phi)*t2;
    return (c*n + s*lateral).normalized();
}

// Novelty-weighted, redundancy-aware mass score (C++14-safe; uses Eigen::Vector3d)
double dirMassScore(
    const Vec3& dir_unit,
    double radius_deg,
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Vec3, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::unordered_map<octomap::OcTreeKey, int,  octomap::OcTreeKey::KeyHash>& hit_count,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    double novelty_pow,                 // e.g., 1.5–1.6
    double redundancy_sigma_deg         // e.g., 12.0
){
    const double cos_thr = std::cos(deg2rad(radius_deg));
    double mass = 0.0;

    // accumulate novelty-weighted potential mass
    std::unordered_map<octomap::OcTreeKey, Vec3, octomap::OcTreeKey::KeyHash>::const_iterator itv;
    for (itv = voxelToSphereMap.begin(); itv != voxelToSphereMap.end(); ++itv) {
        const octomap::OcTreeKey& key = itv->first;
        const Vec3 u = itv->second.normalized();

        if (dir_unit.dot(u) >= cos_thr) {
            float pot = 0.0f;
            try {
                pot = boost::get<float>(model.getVoxelProperty(key, "potential"));
            } catch (...) {
                continue;
            }

            int h = 0;
            std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>::const_iterator ith = hit_count.find(key);
            if (ith != hit_count.end()) h = ith->second;

            // novelty in (0,1], higher for unseen
            const double novelty = std::pow(1.0 / (1.0 + static_cast<double>(h)), novelty_pow);
            mass += static_cast<double>(pot) * novelty;
        }
    }

    // mild anti-redundancy: downscale if too close to existing viewpoints
    if (!viewpoints.empty() && redundancy_sigma_deg > 1e-9) {
        double min_deg = 180.0;
        for (size_t i = 0; i < viewpoints.size(); ++i) {
            const Vec3 v = viewpoints[i]->getPosition().normalized();
            double dotv = dir_unit.dot(v);
            if (dotv < -1.0) dotv = -1.0;
            if (dotv >  1.0) dotv =  1.0;
            const double ang = std::acos(dotv) * 180.0 / M_PI;
            if (ang < min_deg) min_deg = ang;
        }
        double scale = (redundancy_sigma_deg > 0.0) ? (min_deg / redundancy_sigma_deg) : 1.0;
        if (scale < 0.2) scale = 0.2;   // lower bound
        if (scale > 1.0) scale = 1.0;   // upper bound
        mass *= scale;
    }

    return mass;
}

// Local ring refinement of NBV (samples two rings + center; returns point on sphere radius)
Eigen::Vector3d refineNBVLocalRing(
    const Eigen::Vector3d& nbv_pos,
    double sphere_radius,
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>& hit_count,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    double patch_radius_deg_main,   // e.g., 8.0 at ~98% cov; 6.0 near 99%
    double novelty_pow,             // e.g., 1.6
    double redundancy_sigma_deg     // e.g., 12.0
){
    Vec3 center_dir = nbv_pos.normalized();
    Vec3 best_dir   = center_dir;
    double best_score = -1.0;

    const double inner = 0.5 * patch_radius_deg_main;
    const double outer = patch_radius_deg_main;
    const double eval_radius = std::max(6.0, 0.75 * patch_radius_deg_main);

    // center + two rings
    const double radii[3] = {0.0, inner, outer};
    for (int ri = 0; ri < 3; ++ri) {
        double rdeg = radii[ri];
        int steps = (rdeg < 1e-6) ? 1 : 12;
        for (int i = 0; i < steps; ++i) {
            double phi = (2.0*M_PI) * (double(i) / double(steps));
            Vec3 dir = (rdeg < 1e-6) ? center_dir : offsetOnSphere(center_dir, rdeg, phi);
            double sc = dirMassScore(dir, eval_radius, model, voxelToSphereMap, hit_count,
                                     viewpoints, novelty_pow, redundancy_sigma_deg);
            if (sc > best_score) { best_score = sc; best_dir = dir; }
        }
    }
    return best_dir * sphere_radius;
}


int main(int argc, char* argv[]) {
    // -------- setup / IO --------
    int seed = (argc > 1) ? std::atoi(argv[1]) : 0;
    srand(seed);

    const std::string model_path = "../models/cat.stl";   // change as needed
    const std::string model_name = file_stem(model_path);
    const std::string base_dir   = "cpp_output";
    const std::string output_dir = base_dir + "/" + model_name + "/seed_" + std::to_string(seed);
    system(("mkdir -p " + output_dir).c_str());

    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(1.0, 1.0, 1.0));

    visioncraft::Model model;
    std::cout << "Loading model: " << model_path << "\n";
    model.loadModel(model_path, 100000);
    std::cout << "Model loaded successfully.\n";

    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);
    model.addVoxelProperty("potential", 0.0f);

    // -------- PF / sphere / viewpoints --------
    const float  TARGET_COVERAGE = 0.99f;
    const double COV_EPS         = 1e-9;
    const float  sphere_radius   = 400.0f;

    const int start_viewpts  = 3;   // baseline
    const int max_viewpoints = 16;

    auto viewpoints = generateClusteredViewpoints(start_viewpts, sphere_radius);
    for (auto& vp : viewpoints) {
        vp->setDownsampleFactor(8.0);
        visibilityManager->trackViewpoint(vp);
        vp->setFarPlane(900);
        vp->setNearPlane(50);
    }

    // Light pre-spread (repulsion-only)
    {
        float k_rep_pre = 250.0f;
        float sigma_rep_pre = sphere_radius *
            std::acos(std::max(-1.0f, 1.0f - 2.0f / float(std::max(2, int(viewpoints.size())))));
        const int pre_steps = 20;
        for (int s = 0; s < pre_steps; ++s) {
            std::vector<Eigen::Vector3d> delta(viewpoints.size(), Eigen::Vector3d::Zero());
            for (size_t i = 0; i < viewpoints.size(); ++i) {
                Eigen::Vector3d F = computeRepulsiveForce(viewpoints, viewpoints[i], sphere_radius, sigma_rep_pre);
                Eigen::Vector3d n = viewpoints[i]->getPosition().normalized();
                Eigen::Vector3d Ft = F - F.dot(n) * n;
                delta[i] = 1.5 * k_rep_pre * Ft;
            }
            for (size_t i = 0; i < viewpoints.size(); ++i) {
                Eigen::Vector3d new_pos = viewpoints[i]->getPosition() + delta[i];
                updateViewpointState(viewpoints[i], new_pos, sphere_radius, true);
                visualizer.updateViewpoint(*viewpoints[i], false, true, true, true);
            }
        }
    }

    std::unordered_map<int, std::vector<Eigen::Vector3d>> viewpointPaths;
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash> voxelToSphereMap;
    mapVoxelsToSphere(model, sphere_radius, voxelToSphereMap);

    // -------- CSV outputs --------
    std::ofstream csv_file(output_dir + "/results.csv");
    csv_file << "Timestep,CoverageScore\n";
    csv_file << std::fixed << std::setprecision(6);

    std::ofstream time_series_csv(output_dir + "/time_series.csv");
    time_series_csv << "iteration,coverage,redundancy,affinity,time\n";
    time_series_csv << std::fixed << std::setprecision(6);

    std::ofstream viewpoint_csv_file(output_dir + "/viewpoint_data.csv");
    viewpoint_csv_file << "Timestep,ViewpointID,X,Y,Z,OrientationX,OrientationY,OrientationZ,OrientationW\n";
    for (size_t i = 0; i < viewpoints.size(); ++i) {
        auto p = viewpoints[i]->getPosition();
        auto q = viewpoints[i]->getOrientationQuaternion();
        viewpoint_csv_file << "0," << i << "," << p.x() << "," << p.y() << "," << p.z() << ","
                           << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
    }

    // -------- optimizer / PF gains --------
    std::vector<Eigen::Vector3d> m(viewpoints.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> v(viewpoints.size(), Eigen::Vector3d::Zero());
    int   t_adam = 0;

    // Strong attraction, mild repulsion → faster gap hunting
    float k_attr    = 800.0f;
    float k_rep     = 300.0f;
    float sigma_rep = sphere_radius * 2.0f;

    float beta1 = 0.95f, beta2 = 0.999f, lr = 0.5f, adam_eps = 1e-8f;
    float lr_decay = 1.0f;

    // -------- insertion & termination bookkeeping (compact, faster) --------
    bool  phase2 = false;
    int   near_target_streak = 0;

    const int WIN = 8;
    std::deque<float>  recent_moves;
    std::deque<float>  recent_need;
    std::deque<double> recent_cov;

    double initial_total_need = std::max(1.0, double(smoothCoverageNeed(0)) * double(voxelToSphereMap.size()));

    int   max_stage_iterations = 60;  // earlier backstop
    int   iters_since_insert   = 0;
    int   stage_iter           = 0;
    bool  first_insert_done    = false;

    bool   ema_init      = false;
    double ema_cov       = 0.0;
    double ema_gain      = 0.0;
    double ema_gain_abs  = 0.0;
    double ema_move      = 0.0;
    const  double EMA_A  = 0.20;

    // adaptive cooldown + stall thresholds
    auto cooldown_for = [](double cov) {
        if (cov < 0.90)   return 16;
        if (cov < 0.96)   return 22;
        if (cov < 0.985)  return 28;
        return 34;
    };
    auto stall_delta_for = [](double cov) {
        if (cov < 0.90)   return 8e-4;   // 0.08%
        if (cov < 0.96)   return 6e-4;   // 0.06%
        if (cov < 0.985)  return 4e-4;   // 0.04%
        return 2e-4;                     // 0.02%
    };
    auto stall_steps_for = [](double cov) {
        if (cov < 0.90)   return 22;
        if (cov < 0.96)   return 28;
        if (cov < 0.985)  return 36;
        return 44;
    };
    double best_cov = 0.0;
    int    last_improve_iter = 0;

    auto start_time = std::chrono::steady_clock::now();

    // ---- tiny helpers local to main (C++14-safe) ----
    auto deg2rad = [](double d){ return d * M_PI / 180.0; };
    auto tangentBasis = [](const Eigen::Vector3d& n, Eigen::Vector3d& t1, Eigen::Vector3d& t2){
        if (std::fabs(n.z()) < 0.9) t1 = n.cross(Eigen::Vector3d::UnitZ()).normalized();
        else                        t1 = n.cross(Eigen::Vector3d::UnitY()).normalized();
        t2 = n.cross(t1).normalized();
    };
    auto offsetOnSphere = [&](const Eigen::Vector3d& n, double radius_deg, double phi){
        Eigen::Vector3d t1, t2; tangentBasis(n, t1, t2);
        const double th = deg2rad(radius_deg);
        const double c  = std::cos(th), s = std::sin(th);
        Eigen::Vector3d lateral = std::cos(phi)*t1 + std::sin(phi)*t2;
        return (c*n + s*lateral).normalized();
    };
    auto dirMassScore = [&](const Eigen::Vector3d& dir_unit,
                            double radius_deg,
                            double novelty_pow,
                            double redundancy_sigma_deg,
                            const std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>& hit_count)->double {
        const double cos_thr = std::cos(deg2rad(radius_deg));
        double mass = 0.0;

        for (const auto& kv : voxelToSphereMap) {
            const auto& key = kv.first;
            const Eigen::Vector3d u = kv.second.normalized();
            if (dir_unit.dot(u) >= cos_thr) {
                float pot = 0.0f;
                try { pot = boost::get<float>(model.getVoxelProperty(key, "potential")); }
                catch (...) { continue; }
                int h = 0;
                auto it = hit_count.find(key);
                if (it != hit_count.end()) h = it->second;
                const double novelty = std::pow(1.0 / (1.0 + double(h)), novelty_pow);
                mass += double(pot) * novelty;
            }
        }

        if (!viewpoints.empty()) {
            double min_deg = 180.0;
            for (const auto& vp : viewpoints) {
                const Eigen::Vector3d v = vp->getPosition().normalized();
                double dotv = std::max(-1.0, std::min(1.0, dir_unit.dot(v)));
                const double ang = std::acos(dotv) * 180.0 / M_PI;
                if (ang < min_deg) min_deg = ang;
            }
            double scale = (redundancy_sigma_deg > 1e-9) ? (min_deg / redundancy_sigma_deg) : 1.0;
            if (scale < 0.2) scale = 0.2;
            if (scale > 1.0) scale = 1.0;
            mass *= scale;
        }
        return mass;
    };
    auto refineNBVLocalRing = [&](const Eigen::Vector3d& nbv_pos,
                                  double patch_radius_deg_main,
                                  double novelty_pow,
                                  double redundancy_sigma_deg,
                                  const std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>& hit_count)->Eigen::Vector3d {
        Eigen::Vector3d center_dir = nbv_pos.normalized();
        Eigen::Vector3d best_dir   = center_dir;
        double best_score = -1.0;

        std::vector<double> radii = {0.0, 0.5*patch_radius_deg_main, patch_radius_deg_main};
        for (double rdeg : radii) {
            int steps = (rdeg < 1e-6) ? 1 : 12;
            for (int i = 0; i < steps; ++i) {
                double phi = (2.0*M_PI) * (double(i) / double(steps));
                Eigen::Vector3d dir = (rdeg < 1e-6) ? center_dir : offsetOnSphere(center_dir, rdeg, phi);
                double sc = dirMassScore(dir,
                                         std::max(6.0, 0.75*patch_radius_deg_main),
                                         novelty_pow,
                                         redundancy_sigma_deg,
                                         hit_count);
                if (sc > best_score) { best_score = sc; best_dir = dir; }
            }
        }
        return best_dir * sphere_radius;
    };

    // ================= LOOP =================
    for (int iter = 0; iter < 2500; ++iter) {
        ++stage_iter; ++iters_since_insert;

        // 1) visibility (GPU)
        for (auto& vp : viewpoints) vp->performRaycastingOnGPU(model);

        // 2) coverage & stats
        VisStats S = computeVisibilityStats(model, viewpoints);

        // best coverage / stall bookkeeping
        if (S.coverage > best_cov + 1e-12) {
            best_cov = S.coverage;
            last_improve_iter = iter;
        }

        // EMA coverage updates
        double gain_now = 0.0;
        if (!ema_init) {
            ema_init = true;
            ema_cov = S.coverage;
            ema_gain = 0.0;
            ema_gain_abs = 0.0;
            ema_move = 0.0;
        } else {
            gain_now = S.coverage - ema_cov;
            ema_cov += EMA_A * gain_now;
            ema_gain = (1.0 - EMA_A) * ema_gain + EMA_A * gain_now;
            ema_gain_abs = (1.0 - EMA_A) * ema_gain_abs + EMA_A * std::abs(gain_now);
        }

        recent_cov.push_back(S.coverage);
        if (recent_cov.size() > size_t(WIN)) recent_cov.pop_front();

        // 3) potentials
        int current_sample_rate = (S.coverage < 0.90) ? 2 : 1;
        computeVoxelPotentials(model, voxelToSphereMap, viewpoints, sphere_radius,
                               current_sample_rate, S.hit_count);

        float total_potential = 0.0f;
        for (auto it = voxelToSphereMap.begin(); it != voxelToSphereMap.end(); ++it)
            total_potential += boost::get<float>(model.getVoxelProperty(it->first, "potential"));

        // 4) viz
        {
            vtkSmartPointer<vtkPolyData> spherePoly =
                computeInterpolatedPotentialsOnSphere(model, voxelToSphereMap, "potential", sphere_radius);
            float MAX_POT = std::log(M_PI * sphere_radius) * viewpoints.size();
            visualizer.visualizePotentialOnSphere(spherePoly, MAX_POT, 0.5f);
        }

        // 5) logging
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        csv_file        << iter << "," << std::setprecision(4) << S.coverage << "\n";
        time_series_csv << iter << "," << S.coverage << "," << S.redundancy << "," << S.affinity << "," << elapsed << "\n";
        std::cout << "It " << iter
                  << " | cov " << std::fixed << std::setprecision(6) << S.coverage
                  << " | pot " << std::fixed << std::setprecision(2) << total_potential
                  << " | VPs " << viewpoints.size() << "\n";

        // 6) phase switch
        if (S.coverage + COV_EPS >= TARGET_COVERAGE) {
            near_target_streak++;
            if (!phase2 && near_target_streak >= 2) {
                phase2 = true;
                recent_need.clear();
                std::cout << "[Phase-2] Coverage ≥ " << TARGET_COVERAGE << " stabilized. Balancing overlap now.\n";
            }
        } else {
            near_target_streak = 0;
        }

        // 7) need window
        double total_need = computeTotalNeedFromHits(voxelToSphereMap, S.hit_count);
        recent_need.push_back(float(std::fabs(total_need)));
        if (recent_need.size() > size_t(WIN)) recent_need.pop_front();

        // 8) PF relaxation (constant force mix, ADAM)
        t_adam++;
        int inner_relax_steps = (S.coverage < 0.90) ? 6 : 8; // faster decision cadence early

        for (int rr = 0; rr < inner_relax_steps; ++rr) {
            std::vector<Eigen::Vector3d> steps(viewpoints.size(), Eigen::Vector3d::Zero());
            std::vector<std::thread> threads;
            size_t idx = 0;

            for (auto& vp : viewpoints) {
                threads.emplace_back([&, idx, vp]() {
                    Eigen::Vector3d F_attr = computeAttractiveForce(
                        model, voxelToSphereMap, S.hit_count, vp, sphere_radius, current_sample_rate
                    );
                    Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, vp, sphere_radius, sigma_rep);
                    Eigen::Vector3d F_total = k_attr * F_attr + k_rep * F_repel;

                    Eigen::Vector3d n = vp->getPosition().normalized();
                    Eigen::Vector3d Ft = F_total - F_total.dot(n) * n;

                    steps[idx] = computeStep(Ft, m[idx], v[idx], t_adam, lr, beta1, beta2, adam_eps, true);
                });
                ++idx;
            }
            for (auto& th : threads) th.join();

            float avg_move = 0.0f;
            for (size_t i = 0; i < viewpoints.size(); ++i) {
                Eigen::Vector3d new_pos = viewpoints[i]->getPosition() + steps[i];
                updateViewpointState(viewpoints[i], new_pos, sphere_radius, !phase2);
                visualizer.updateViewpoint(*viewpoints[i], false, true, true, true);
                avg_move += float(steps[i].norm());
            }
            if (!viewpoints.empty()) avg_move /= float(viewpoints.size());
            recent_moves.push_back(avg_move);
            if (recent_moves.size() > size_t(WIN)) recent_moves.pop_front();

            if (ema_move == 0.0) ema_move = avg_move;
            else                  ema_move = (1.0 - EMA_A) * ema_move + EMA_A * avg_move;
        }

        // 9) NBV by potential mass (slightly wider mid-band to prep a big 6th)
        double ang_deg = (S.coverage < 0.93)   ? 18.0 :
                         (S.coverage < 0.965)  ? 16.0 :
                         (S.coverage < 0.985)  ? 12.0 : 10.0;
        int    topK    = 400;
        NBVMassResult nbv_mass = pickNBVByPotentialMass(model, voxelToSphereMap, sphere_radius, topK, ang_deg);
        Eigen::Vector3d nbv_pos = nbv_mass.pos;

        // Late-stage micro-refinement to squeeze the marginal 6th–7th gains
        if (S.coverage >= 0.975) {
            double patch = (S.coverage < 0.985) ? 8.0 : 6.0;
            nbv_pos = refineNBVLocalRing(nbv_pos,
                                         /*patch_radius_deg_main=*/patch,
                                         /*novelty_pow=*/1.6,
                                         /*redundancy_sigma_deg=*/12.0,
                                         S.hit_count);
        }

        // 10) insertion decision — adaptive, non-greedy, biased to 7–8 VPs
        bool windows_ready = (recent_moves.size() == size_t(WIN)) && (recent_need.size() == size_t(WIN));
        bool unmet_target  = (S.coverage + COV_EPS < TARGET_COVERAGE);

        const int min_insert_cooldown = cooldown_for(S.coverage);
        bool cooldown_ok = (first_insert_done ? (iters_since_insert >= min_insert_cooldown) : true);

        float avg_move_w = 0.0f;
        if (!recent_moves.empty())
            avg_move_w = std::accumulate(recent_moves.begin(), recent_moves.end(), 0.0f) / float(recent_moves.size());
        bool motion_plateau = (ema_move > 0.0) ? (avg_move_w <= 0.85f * float(ema_move)) : false;

        float need_rate_norm = 1000.0f;
        if (recent_need.size() >= 2) {
            float start = recent_need.front();
            float end   = recent_need.back();
            float slope = (end - start) / std::max(1.0f, float(int(recent_need.size()) - 1));
            need_rate_norm = float(slope / std::max(1e-9f, float(initial_total_need)));
        }
        float mean_need = 0.0f;
        for (float x : recent_need) mean_need += x;
        mean_need /= std::max<size_t>(1, recent_need.size());
        bool need_plateau = (std::fabs(need_rate_norm) <= 0.08f * std::max(1e-6f, mean_need / float(WIN)));

        bool coverage_starved = (ema_gain_abs > 0.0) ? (std::abs(ema_gain) <= 0.25 * ema_gain_abs) : false;

        // Soft cap tuned to prefer 7–8
        int  soft_cap = (S.coverage < 0.90)   ? std::min(6,  max_viewpoints)
                     : (S.coverage < 0.965)  ? std::min(7,  max_viewpoints)
                     : (S.coverage < 0.985)  ? std::min(8,  max_viewpoints)
                                             : std::min(9,  max_viewpoints);
        bool under_soft_cap = (int(viewpoints.size()) < soft_cap);

        const double stall_delta  = stall_delta_for(S.coverage);
        const int    stall_steps  = stall_steps_for(S.coverage);
        bool stalled = (iter - last_improve_iter >= stall_steps) &&
                       ((best_cov - ema_cov) <= stall_delta);

        bool insert = windows_ready &&
                      unmet_target &&
                      cooldown_ok &&
                      under_soft_cap &&
                      (motion_plateau || coverage_starved || need_plateau);

        if (!insert && unmet_target && cooldown_ok && under_soft_cap && stalled)
            insert = true;

        if (!insert && windows_ready && (stage_iter > max_stage_iterations) && cooldown_ok)
            insert = true;

        if (insert && int(viewpoints.size()) < max_viewpoints) {
            addNewViewpoint(viewpoints, visibilityManager, visualizer,
                            nbv_pos, Eigen::Vector3d(0,0,0), sphere_radius);

            // adapt σ_rep to N
            sigma_rep = sphere_radius *
                        std::acos(std::max(-1.0, 1.0 - 2.0 / double(std::max(2, int(viewpoints.size())))));

            // late VP tweaks to dig deeper into tiny gaps
            if (S.coverage >= 0.975) {
                viewpoints.back()->setDownsampleFactor(4.0); // finer rays only for late VPs
                viewpoints.back()->setNearPlane(30);         // see tighter concavities
                // sigma_rep *= 0.999;                           // allow slightly closer packing near the end
            }

            // expand optimizer states
            m.push_back(Eigen::Vector3d::Zero());
            v.push_back(Eigen::Vector3d::Zero());
            t_adam = 0;

            // reset windows / counters
            recent_moves.clear();
            recent_need.clear();
            recent_cov.clear();
            stage_iter = 0;
            iters_since_insert = 0;
            first_insert_done = true;

            viewpointPaths[viewpoints.size() - 1] = {viewpoints.back()->getPosition()};
            std::cout << "[Insert] NBV mass insert. N=" << viewpoints.size()
                      << " | mass=" << std::fixed << std::setprecision(3) << nbv_mass.mass
                      << " | r=" << nbv_mass.radius_deg << "deg"
                      << " | cooldown=" << min_insert_cooldown
                      << " | stall=" << std::boolalpha << stalled << "\n";
        }

        // 11) trails + render
        for (size_t i = 0; i < viewpoints.size(); ++i)
            viewpointPaths[i].push_back(viewpoints[i]->getPosition());
        visualizer.addVoxelMapProperty(model, "visibility",
                                       Eigen::Vector3d(1,0,0), Eigen::Vector3d(0,1,0));
        visualizer.visualizePaths(viewpointPaths, sphere_radius);
        visualizer.render();
        visualizer.removeVoxelMapProperty();

        // 12) write VP states this iter
        for (size_t i = 0; i < viewpoints.size(); ++i) {
            auto p = viewpoints[i]->getPosition();
            auto q = viewpoints[i]->getOrientationQuaternion();
            viewpoint_csv_file << iter << "," << i << "," << p.x() << "," << p.y() << "," << p.z() << ","
                               << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        }

        // 13) termination
        if (S.coverage + COV_EPS >= TARGET_COVERAGE) {
            std::cout << "[Stop] target coverage reached (" << std::setprecision(6) << S.coverage << ")\n";
            break;
        }

        lr *= lr_decay;
    }

    // -------- finalize --------
    viewpoint_csv_file.close();
    csv_file.close();
    time_series_csv.close();

    std::cout << "Total viewpoints selected: " << viewpoints.size() << std::endl;
    exportFinalMetrics(model, viewpoints, output_dir, seed);
    return 0;
}
