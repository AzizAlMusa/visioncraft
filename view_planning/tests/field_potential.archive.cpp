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
#include <open3d/Open3D.h>

// Generate viewpoints clustered near a specific region
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


// Helper function to compute the geodesic distance between two points on a sphere
float computeGeodesicDistance(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, float sphere_radius)
{
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


float computeSigmoid(float x){
    float sigmoid_input = (x - 1.0f) * 5.0f;
    return 1.0 / (1.0 + std::exp(-sigmoid_input));
}

// Hash function for Eigen::Vector3d
struct Vector3dHash {
    std::size_t operator()(const Eigen::Vector3d& vec) const {
        std::size_t h1 = std::hash<double>{}(vec.x());
        std::size_t h2 = std::hash<double>{}(vec.y());
        std::size_t h3 = std::hash<double>{}(vec.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};



// Function to map voxels to sphere positions
// Function to map voxels to sphere positions
void mapVoxelsToSphere(
    visioncraft::Model& model,
    float sphere_radius,
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap
)
{
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

    double voxelSize = model.getVoxelSize(); // Assume this returns the edge length of the voxel cube
    double diagonalLength = voxelSize * std::sqrt(3.0); // Diagonal length of the voxel cube

    std::vector<Eigen::Vector3d> unoccludedPositions;
    //create a map of unoccluded positions to their keys
    std::unordered_map<Eigen::Vector3d, octomap::OcTreeKey, Vector3dHash> unoccludedPositionsToKeys;
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash> selfOccludedVoxels;

    int numUnoccluded = 0;
    int numSelfOccluded = 0;

    // For each voxel
    for (const auto& kv : voxelMap) {
        const auto& voxel = kv.second;
        const auto& key = kv.first;

        Eigen::Vector3d voxelPosition = voxel.getPosition();
        Eigen::Vector3d normal;
        try {
            normal = boost::get<Eigen::Vector3d>(model.getVoxelProperty(key, "normal"));
        } catch (const boost::bad_get&) {
            std::cerr << "[WARNING] Voxel at key " << key.k[0] << ", " << key.k[1] << ", " << key.k[2]
                      << " does not have a normal property. Skipping." << std::endl;
            continue;
        }
        normal.normalize(); // Ensure the normal is unit length

        // Compute the intersection point of the ray from voxelPosition along normal with the sphere
        double a = 1.0; // normal is normalized
        double b = 2.0 * voxelPosition.dot(normal);
        double c = voxelPosition.squaredNorm() - sphere_radius * sphere_radius;

        double discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0) {
            std::cerr << "[DEBUG] No intersection for voxel at " << voxelPosition.transpose() << std::endl;
            continue;
        }

        double sqrt_disc = std::sqrt(discriminant);
        double t1 = (-b + sqrt_disc) / (2.0 * a);
        double t2 = (-b - sqrt_disc) / (2.0 * a);

        double t = std::max(t1, t2); // Choose the larger t to ensure the ray goes outward
        if (t <= 0) {
            std::cerr << "[DEBUG] Intersection is behind the voxel at " << voxelPosition.transpose() << std::endl;
            continue;
        }

        Eigen::Vector3d spherePoint = voxelPosition + t * normal;

        // Adjust the starting point of the ray to avoid self-occlusion
        Eigen::Vector3d rayStart = voxelPosition + diagonalLength * normal; // Start raycasting after the diagonal

        bool occluded = false;
        octomap::point3d origin(rayStart.x(), rayStart.y(), rayStart.z());
        octomap::point3d direction(normal.x(), normal.y(), normal.z());
        double maxRange = t - diagonalLength; // Reduce range to account for starting offset

        octomap::point3d end;
        bool hit = octree->castRay(origin, direction, end, true, maxRange);

        if (hit) {
            std::cerr << "[DEBUG] Voxel at " << voxelPosition.transpose() 
                      << " is occluded by another voxel at " << end << std::endl;
            occluded = true;
            numSelfOccluded++;
        }

        if (!occluded) {
            voxelToSphereMap[key] = spherePoint;
            unoccludedPositions.push_back(voxelPosition);
            unoccludedPositionsToKeys[voxelPosition] = key;
            numUnoccluded++;
        } else {
            selfOccludedVoxels[key] = spherePoint;
        }
    }

    std::cout << "[INFO] Number of unoccluded voxels: " << numUnoccluded << std::endl;
    std::cout << "[INFO] Number of self-occluded voxels: " << numSelfOccluded << std::endl;

    // Use KD-tree for finding the nearest unoccluded neighbor for self-occluded voxels
    if (!unoccludedPositions.empty()) {
        auto kdtree = std::make_shared<open3d::geometry::KDTreeFlann>();
        auto pointCloud = std::make_shared<open3d::geometry::PointCloud>();
        pointCloud->points_ = unoccludedPositions;
        kdtree->SetGeometry(*pointCloud);

        int numMapped = 0;

        for (const auto& kv : selfOccludedVoxels) {
            const auto& key = kv.first;
            Eigen::Vector3d voxelPosition = model.getVoxel(key)->getPosition();

            std::vector<int> indices;
            std::vector<double> distances;

            if (kdtree->SearchKNN(voxelPosition, 1, indices, distances) > 0) {
                int nearestIdx = indices[0];
                auto nearestVoxelPosition = unoccludedPositions[nearestIdx];
                auto nearestVoxelKey = unoccludedPositionsToKeys[nearestVoxelPosition];
                Eigen::Vector3d nearestSpherePoint = voxelToSphereMap[nearestVoxelKey];
                voxelToSphereMap[key] = nearestSpherePoint;
                numMapped++;
            } else {
                std::cerr << "[WARNING] No nearest unoccluded neighbor found for self-occluded voxel at "
                          << voxelPosition.transpose() << std::endl;
            }
        }
        std::cout << "[INFO] Number of self-occluded voxels mapped to nearest neighbors: " << numMapped << std::endl;
    } else {
        std::cerr << "[ERROR] No unoccluded positions available to map self-occluded voxels." << std::endl;
    }

    std::cout << "[INFO] Total voxels mapped to sphere: " << voxelToSphereMap.size() << std::endl;
}

// // Compute potential for each voxel based on distances to all viewpoints
// void computeVoxelPotentials(
//     visioncraft::Model& model,
//     const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
//     int V_max,
//     bool use_exponential,
//     float sigma = 1.0f) // Default value; adjust as needed
// {
//     const auto& voxelMap = model.getVoxelMap().getMap();

//     // Loop over each voxel in the map
//     for (const auto& kv : voxelMap) {
//         const auto& voxel = kv.second;
//         const auto& key = kv.first;
        
//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         float V_norm = static_cast<float>(visibility) / V_max;

//         // V_norm > 1.0f ? V_norm = 1.0f : V_norm;

//         float potential = 0.0f;
//         for (const auto& viewpoint : viewpoints) {
//             Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition() ;
//             float distance_squared = r.squaredNorm();

//             if (use_exponential) {
//                 potential += std::exp(-distance_squared / (2.0f * sigma * sigma));
//                 model.setVoxelProperty(key, "potential", potential);
//             } else {
//                 potential += distance_squared;
//             }
//         } 
//         model.setVoxelProperty(key, "potential",  (1.0f - V_norm) * potential);
//         // std::cout << "Potential: " << (1.0f - V_norm) * potential << std::endl;
//     }


// }

// Compute potential for each voxel based on geodesic distances to all viewpoints
// void computeVoxelPotentials(
//     visioncraft::Model& model,
//     const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
//     const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
//     float sphere_radius,
//     int V_max,
//     bool use_exponential,
//     float sigma = 1.0f) // Default value; adjust as needed
// {
//     // Loop over each voxel in the map
    
//     for (const auto& kv : voxelToSphereMap) {
//         const auto& key = kv.first;
//         const Eigen::Vector3d& voxel_position = kv.second;

//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         float V_norm = static_cast<float>(visibility) / V_max;

//         float potential = 0.0f;

//         for (const auto& viewpoint : viewpoints) {
//             Eigen::Vector3d viewpoint_position = viewpoint->getPosition().normalized() * sphere_radius;

//             float geodesic_distance = computeGeodesicDistance(viewpoint_position, voxel_position, sphere_radius);

//             potential += (1 - computeSigmoid(visibility) ) * std::exp(-geodesic_distance * geodesic_distance / (2.0f * sigma * sigma));
//         }

//         model.setVoxelProperty(key, "potential", potential);
//         // std::cout << "Potential: " << potential << std::endl;
//     }
// }

void computeVoxelPotentials(
    visioncraft::Model& model,
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    float sphere_radius,
    int V_max,
    bool use_exponential,
    float sigma = 1.0f
) {
    float epsilon = 1e-6;  // Prevents division by zero

    for (const auto& kv : voxelToSphereMap) {
        const auto& key = kv.first;
        const Eigen::Vector3d& voxel_position = kv.second;
        const Eigen::Vector3d& sphere_position = voxelToSphereMap[key];

        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / V_max;

        float potential = 0.0f;

        for (const auto& viewpoint : viewpoints) {
            Eigen::Vector3d viewpoint_position = viewpoint->getPosition();

            float geodesic_distance = computeGeodesicDistance(viewpoint_position, sphere_position, sphere_radius);

            potential += (1.0f - computeSigmoid(visibility)) * std::log(geodesic_distance + epsilon);
        }

        model.setVoxelProperty(key, "potential", potential);
    }
}



// Eigen::Vector3d computeAttractiveForce(
//     visioncraft::Model& model,
//     const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
//     const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
//     float sphere_radius,
//     float sigma,
//     int max_visibility,
//     bool use_exponential = false)
// {
//     Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
//     Eigen::Vector3d viewpoint_position = viewpoint->getPosition().normalized() * sphere_radius;

//     for (const auto& kv : voxelToSphereMap) {
//         const auto& key = kv.first;
//         const Eigen::Vector3d& voxel_position = kv.second;

//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         float V_norm = static_cast<float>(visibility) / max_visibility;

//         float geodesic_distance = computeGeodesicDistance(viewpoint_position, voxel_position, sphere_radius);
//         if (geodesic_distance < 1e-8) continue;

//         // Compute direction of geodesic tangent vector
//         Eigen::Vector3d d = voxel_position - viewpoint_position;
//         Eigen::Vector3d tangent_vector = d - (d.dot(viewpoint_position.normalized())) * viewpoint_position.normalized();
//         tangent_vector.normalize();

//         // Compute weight based on visibility and distance
//         float weight = (1 - computeSigmoid(visibility)) * geodesic_distance / (sigma * sigma) *
//                        std::exp(-geodesic_distance * geodesic_distance / (2.0f * sigma * sigma));

//         total_force += weight * tangent_vector  * 1000.0f;
        
//         float voxel_force = static_cast<float>(((weight * tangent_vector * 1000.0f).norm()));
//         std::cout << "Voxel force: " << voxel_force << std::endl;
//         model.setVoxelProperty(key, "force",  voxel_force);
        
  
//     }

//     return total_force ;
// }

Eigen::Vector3d computeAttractiveForce(
    const visioncraft::Model& model,
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    float sphere_radius,
    float sigma,
    int max_visibility,
    bool use_exponential = false
) {
    float epsilon = 1e-6;  // Prevents division by zero
    Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d viewpoint_position = viewpoint->getPosition();

    for (const auto& kv : voxelToSphereMap) {
        const auto& key = kv.first;
        const Eigen::Vector3d& voxel_position = kv.second;
        const Eigen::Vector3d& sphere_position = voxelToSphereMap[key];

        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / max_visibility;

        float geodesic_distance = computeGeodesicDistance(viewpoint_position, sphere_position, sphere_radius);
        if (geodesic_distance < 1e-8) continue;

        // Compute tangent vector along the geodesic
        Eigen::Vector3d d = sphere_position - viewpoint_position;
        Eigen::Vector3d tangent_vector = d - (d.dot(viewpoint_position.normalized())) * viewpoint_position.normalized();
        tangent_vector.normalize();

        // Compute force contribution
        float weight = (1.0f - computeSigmoid(visibility)) / (geodesic_distance + epsilon);

        total_force += weight * tangent_vector *1000.0f;
    }

    return total_force;
}



// Eigen::Vector3d computeAttractiveForce(
//     const visioncraft::Model& model,
//     const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
//     float sigma,
//     int V_max)
// {   

//     Eigen::Vector3d F_attr = Eigen::Vector3d::Zero();
//     const auto& voxelMap = model.getVoxelMap().getMap();

//     for (const auto& kv : voxelMap) {
//         const auto& voxel = kv.second;
//         const auto& key = kv.first;

//         // Check if voxel potential was computed
//         float potential = boost::get<float>(model.getVoxelProperty(key, "potential"));
        


//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         float V_norm = static_cast<float>(visibility) / V_max;

//         // V_norm > 1.0f ? V_norm = 1.0f : V_norm;

//         Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();

//         // Gradient of the potential with respect to the viewpoint position
//         Eigen::Vector3d grad_U = 2 * (1.0f - V_norm) * r ;

//         F_attr -= grad_U / sigma;
    
//     }

//     return F_attr;
// }

// Eigen::Vector3d computeAttractiveForce(
//     const visioncraft::Model& model,
//     const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
//     float sigma,
//     int V_max)
// {
//     Eigen::Vector3d F_attr = Eigen::Vector3d::Zero();
//     const auto& voxelMap = model.getVoxelMap().getMap();
//     float sigma_squared = sigma * sigma;

//     for (const auto& kv : voxelMap) {
//         const auto& voxel = kv.second;
//         const auto& key = kv.first;

//         // Retrieve potential directly from the voxel properties
//         float potential = boost::get<float>(model.getVoxelProperty(key, "potential"));

//         // Retrieve visibility and normalize it
//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         float V_norm = static_cast<float>(visibility) / V_max;

//         // Compute vector from viewpoint to voxel
//         Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();
//         float distance_squared = r.squaredNorm();

//         // Exponential term for distance weighting
//         float exp_term = std::exp(-distance_squared / (2.0f * sigma_squared));

//         // Compute gradient of potential with respect to viewpoint position
//         Eigen::Vector3d grad_U =  (1.0f - V_norm) * exp_term * r / sigma_squared;

//         // Accumulate the attractive force
//         F_attr -= grad_U;
//     }

//     return F_attr;
// }





Eigen::Vector3d computeRepulsiveForce(
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    const std::shared_ptr<visioncraft::Viewpoint>& current_viewpoint,
    float sphere_radius,
    float k_repel,
    float alpha
) {
    float epsilon = 1e-6;  // Prevent division by zero
    Eigen::Vector3d total_repulsive_force = Eigen::Vector3d::Zero();

    // Use the given position of the current viewpoint
    Eigen::Vector3d current_position = current_viewpoint->getPosition();

    // Iterate through all other viewpoints
    for (const auto& other_viewpoint : viewpoints) {
        if (current_viewpoint != other_viewpoint) {
            // Use the given position of the other viewpoint
            Eigen::Vector3d other_position = other_viewpoint->getPosition();

            // Compute the geodesic distance
            float geodesic_distance = computeGeodesicDistance(current_position, other_position, sphere_radius);
            if (geodesic_distance < epsilon) {
                continue;  // Skip near-zero distances to avoid singularities
            }

            // Compute the tangent vector for the repulsive force
            Eigen::Vector3d d = other_position - current_position;
            Eigen::Vector3d tangent_vector = d - (d.dot(current_position.normalized())) * current_position.normalized();
            tangent_vector.normalize();

            // Compute the magnitude of the repulsive force
            float force_magnitude = k_repel / std::pow(geodesic_distance, alpha );

            // Accumulate the total repulsive force
            total_repulsive_force -= force_magnitude * tangent_vector;
        }
    }

    return total_repulsive_force;
}



// Update viewpoint state
void updateViewpointState(
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    const Eigen::Vector3d& new_position,
    float sphere_radius) 
{
    Eigen::Vector3d normalized_position = sphere_radius * new_position.normalized();
    viewpoint->setPosition(normalized_position);
    viewpoint->setLookAt(Eigen::Vector3d(0.0, 0.0, 0.0), -Eigen::Vector3d::UnitZ());
}



double computeCoverageScoreChange(double current_score, double previous_score) {
    return std::abs(current_score - previous_score);
}


// Function to compute the total system energy
double computeSystemEnergy(
    const visioncraft::Model& model,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    float sigma,
    int V_max,
    float k_repel,
    float alpha) 
{
    double energy = 0.0;
    const auto& voxelMap = model.getVoxelMap().getMap();

    // Compute voxel potential energy
    for (const auto& kv : voxelMap) {
        const auto& key = kv.first;
        float potential = boost::get<float>(model.getVoxelProperty(key, "potential"));
        energy += potential;
    }

    // // Compute repulsive energy between viewpoints
    for (size_t i = 0; i < viewpoints.size(); ++i) {
        for (size_t j = i + 1; j < viewpoints.size(); ++j) {
            Eigen::Vector3d r = viewpoints[i]->getPosition() - viewpoints[j]->getPosition();
            double distance = r.norm() + 1e-5;
            energy += k_repel / std::pow(distance, alpha);
        }
    }

    return energy;
}

// Function to compute the kinetic energy of each viewpoint
double computeKineticEnergy(
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    const std::vector<Eigen::Vector3d>& previous_positions,
    float delta_t,
    float mass = 1.0f, // Default mass is 1.0
    bool return_average = false) // If true, returns the average kinetic energy
{
    double total_kinetic_energy = 0.0;

    for (size_t i = 0; i < viewpoints.size(); ++i) {
        Eigen::Vector3d velocity = (viewpoints[i]->getPosition() - previous_positions[i]) / delta_t;


        double speed_squared = velocity.squaredNorm(); // v^2
        total_kinetic_energy += 0.5 * mass * speed_squared; // KE = 1/2 m v^2
    }

    if (return_average) {
        return total_kinetic_energy / viewpoints.size(); // Average kinetic energy
    } else {
        return total_kinetic_energy; // Total kinetic energy
    }
}


double computeAverageForce(
    const visioncraft::Model& model,
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    float sphere_radius,
    float k_repel,
    float alpha,
    float sigma,
    int V_max)
{
    double total_force_magnitude = 0.0;

    for (const auto& viewpoint : viewpoints) {
        Eigen::Vector3d F_attr = computeAttractiveForce(const_cast<visioncraft::Model&>(model), voxelToSphereMap, viewpoint, sphere_radius, sigma, V_max);
        
        Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, sphere_radius, k_repel, alpha);
        Eigen::Vector3d F_total = F_attr + F_repel;

        total_force_magnitude += F_total.norm(); // Magnitude of the total force
    }

    return total_force_magnitude / viewpoints.size(); // Average force magnitude
}


double computeEntropy(const visioncraft::Model& model)
{
    double entropy = 0.0;
    const auto& voxelMap = model.getVoxelMap().getMap();
    double total_visibility = 0.0;

    // Step 1: Compute total visibility
    for (const auto& kv : voxelMap) {
        int visibility = boost::get<int>(model.getVoxelProperty(kv.first, "visibility"));
        total_visibility += static_cast<double>(visibility);
    }

    // Step 2: Compute entropy
    for (const auto& kv : voxelMap) {
        int visibility = boost::get<int>(model.getVoxelProperty(kv.first, "visibility"));

        if (visibility > 0) {
            double probability = static_cast<double>(visibility) / total_visibility;
            entropy -= probability * std::log(probability); // Add -p * log(p)
        }
    }

    return entropy;
}


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






int main() {
    srand(time(nullptr));

    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    visioncraft::Model model;
    
    std::cout << "Loading model..." << std::endl;
    model.loadModel("../models/cat.ply", 100000);
    std::cout << "Model loaded successfully." << std::endl;

    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);
    model.addVoxelProperty("potential", 0.0f);
    model.addVoxelProperty("force", 0.0f);

    float sphere_radius = 400.0f;
    int num_viewpoints = 8;
    auto viewpoints = generateClusteredViewpoints(num_viewpoints, sphere_radius);

    for (auto& viewpoint : viewpoints) {
        viewpoint->setDownsampleFactor(8.0);
        visibilityManager->trackViewpoint(viewpoint);
        viewpoint->setFarPlane(900);
        viewpoint->setNearPlane(300);
        
    }
    std::unordered_map<int, std::vector<Eigen::Vector3d>> viewpointPaths;

    // Simulation parameters
    float sigma = 1000.0f;
    float k_attr = 1000.0f;
    float k_repel = 50000.0f;
    float delta_t = 0.04f;
    float alpha = 1.0f;
    int max_iterations = 100;
    int V_max = num_viewpoints; //num_viewpoints

    // Generate manifold mapping
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash> voxelToSphereMap;
    mapVoxelsToSphere(model, sphere_radius, voxelToSphereMap);

  
    // Prepare CSV logging
    std::ofstream csv_file("results.csv");
    csv_file << "Timestep,CoverageScore,SystemEnergy,KineticEnergy,ForceMagnitude,Entropy\n";
    csv_file << std::fixed << std::setprecision(6);
    
    // Add this to initialize the CSV file for viewpoint positions
    std::ofstream viewpoint_csv_file("viewpoint_positions.csv");

    // Write header for the viewpoint CSV file
    viewpoint_csv_file << "Timestep,ViewpointID,X,Y,Z\n";

    // Metrics tracking
    double previous_coverage_score = 0.0;

    std::vector<Eigen::Vector3d> previous_positions;
    for (const auto& viewpoint : viewpoints) {
        previous_positions.push_back(viewpoint->getPosition());
    }

    // visualizer.addVoxelMapProperty(model, "visibility");
    // //iterate through voxel keys and visualize the voxel to sphere mapping
    // for (const auto& kv : voxelToSphereMap) {
    //     const auto& key = kv.first;
    //     const auto& spherePoint = kv.second;
    //     // std::cout << "Key: " << key.k[0] << ", " << key.k[1] << ", " << key.k[2] << std::endl;
    //     // std::cout << "Sphere Point: " << spherePoint.transpose() << std::endl;
    //     visualizer.visualizeVoxelToSphereMapping(model, key, voxelToSphereMap);

            
    //     // Eigen::Vector3d normalColor(1.0, 1.0, 1.0); // Red color for the normals
    //     // double normalLength = 10.0; // Length of the normal vector

    //     // visualizer.visualizeVoxelNormals(model, normalLength, normalColor, key);
    //     // visualizer.render();
    //     // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // }
    // while loop that terminates when pressing the 'q' key
    // while (true) {
    //     visualizer.render();
    //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // }

    // loop through angles of a circle and make points on the sphere

        
    Eigen::Vector3d position(400.0f, 0.0f, 0.0f);
    Eigen::Vector3d look_at(0.0, 0.0, 0.0);

    auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, look_at);

    // #include <random>
    // for (int i = 0; i < 100; i++) {
    //     float sphere_radius = 400.0f;

    //     // Create random number generators for theta and phi
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<float> dist_theta(0.0f, 2 * M_PI);  // theta from 0 to 2π
    //     std::uniform_real_distribution<float> dist_phi(0.0f, M_PI);         // phi from 0 to π

    //     // Generate random theta and phi
    //     float theta = dist_theta(gen);
    //     float phi = dist_phi(gen);

    //     // Convert spherical coordinates to Cartesian coordinates
    //     float x = sphere_radius * sin(phi) * cos(theta);
    //     float y = sphere_radius * sin(phi) * sin(theta);
    //     float z = sphere_radius * cos(phi);  // z is calculated from phi

    //     Eigen::Vector3d spherePoint(x, y, z);

    //     // Show geodesic and calculate distance
    //     visualizer.showGeodesic(*viewpoint, spherePoint, sphere_radius);
    //     auto geo_dist = computeGeodesicDistance(viewpoint->getPosition(), spherePoint, sphere_radius);
    //     std::cout << "Geodesic distance: " << geo_dist << std::endl;

    //     // Render and wait for the next iteration
    //     visualizer.render();
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }


    //generate random point on the sphere
    // Eigen::Vector3d spherePoint(-399.0f, 0.0f, 0.0f);  // Example point on the sphere
    
    // Eigen::Vector3d position(400.0f, 0.0f, 0.0f);
    // Eigen::Vector3d look_at(0.0, 0.0, 0.0);

    // auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, look_at);
    // get the first viewpoint
    
    // visualizer.showGeodesic(*viewpoint, spherePoint, sphere_radius);  // Visualize the geodesic curve
    // computeGeodesicDistance(viewpoint->getPosition(), spherePoint, sphere_radius);  // Compute the geodesic distance
    // std::cout << "Geodesic distance: " << computeGeodesicDistance(viewpoint->getPosition(), spherePoint, sphere_radius) << std::endl;
  
   

    for (int iter = 0; iter < max_iterations; ++iter) {

        

        // Perform raycasting
        for (auto& viewpoint : viewpoints) {
            viewpoint->performRaycastingOnGPU(model);
        }
  
        // Log viewpoint positions to the CSV file
        for (size_t i = 0; i < viewpoints.size(); ++i) {
            Eigen::Vector3d position = viewpoints[i]->getPosition();
            viewpoint_csv_file << iter << "," << i << ","
                            << position.x() << "," << position.y() << "," << position.z() << "\n";
            
            viewpointPaths[i].push_back(viewpoints[i]->getPosition());

        }

        bool use_exponential = false; // Set to false for quadratic potential

        computeVoxelPotentials(model, voxelToSphereMap, viewpoints, sphere_radius, V_max, use_exponential, sigma);

        visualizer.visualizePotentialOnSphere(model, sphere_radius, "potential", voxelToSphereMap);

        // Compute metrics
        double coverage_score = visibilityManager->computeCoverageScore();
  
       // Compute system energy
        double system_energy = computeSystemEnergy(model, viewpoints, sigma, V_max, k_repel, alpha);
        double kinetic_energy = computeKineticEnergy(viewpoints, previous_positions, delta_t);
        double average_force = computeAverageForce(model, voxelToSphereMap, viewpoints, sphere_radius, k_repel, alpha, sigma, V_max);
        double system_entropy = computeEntropy(model);


        // Log metrics to CSV
        csv_file << iter << "," << coverage_score << "," << system_energy << "," << kinetic_energy << "," << average_force << "," << system_entropy <<"\n";
        
        // Print metrics to console
        std::cout << "Iteration: " << iter 
                  << ", Coverage Score: " << coverage_score 
                  << ", System Energy: " << system_energy
                  << ", Kinetic Energy: " << kinetic_energy 
                  << ", Average Force: " << average_force 
                  << ", Entropy: " << system_entropy <<"\n";

         for (size_t i = 0; i < viewpoints.size(); ++i) {
            // Update previous positions
            previous_positions[i] = viewpoints[i]->getPosition();
        }

        // Update viewpoint positions
        for (auto& viewpoint : viewpoints) {
         

            // Eigen::Vector3d F_attr = computeAttractiveForce(model, viewpoint, sphere_radius, sigma, num_viewpoints);
            Eigen::Vector3d F_attr = computeAttractiveForce(const_cast<visioncraft::Model&>(model), voxelToSphereMap, viewpoint, sphere_radius, sigma, V_max, use_exponential);
            Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, sphere_radius, k_repel, alpha);
            // std::cout << "F_attr: " << F_attr.transpose() << "F_repel: " << F_repel.transpose() << std::endl;
 

            Eigen::Vector3d F_total = F_attr + F_repel ; // + F_repel
            Eigen::Vector3d n = viewpoint->getPosition().normalized();
            Eigen::Vector3d F_tangent = F_total - F_total.dot(n) * n;
            // compute the percentage of f_tanget over the total force
            double percentage = F_tangent.norm() / F_total.norm();
            // std::cout << "percentage: " << percentage << std::endl;

            // std::cout << "F_tangent: " << F_tangent.transpose() << std::endl;
            Eigen::Vector3d new_position = viewpoint->getPosition() + delta_t * F_tangent;
            updateViewpointState(viewpoint, new_position, sphere_radius);

            // visualizer.addViewpoint(*viewpoint, false, true);
            visualizer.updateViewpoint(*viewpoint, false, true, true, true);
        }

        visualizer.addVoxelMapProperty(model, "visibility");
        visualizer.visualizePaths(viewpointPaths, sphere_radius);

        visualizer.render();
        // visualizer.removeViewpoints();
        visualizer.removeVoxelMapProperty();

        std::this_thread::sleep_for(std::chrono::milliseconds(1));


        // if (iter % 10 == 0 && iter != 0) { // Add a new viewpoint every 10 iterations
        //     Eigen::Vector3d random_position(
        //         static_cast<float>(rand()) / RAND_MAX * sphere_radius,
        //         static_cast<float>(rand()) / RAND_MAX * sphere_radius,
        //         static_cast<float>(rand()) / RAND_MAX * sphere_radius);
        //     Eigen::Vector3d look_at(0.0, 0.0, 0.0);

        //     // Add the new viewpoint
        //     addNewViewpoint(viewpoints, visibilityManager, visualizer, random_position, look_at, sphere_radius);

        //     // Initialize its previous position
        //     previous_positions.push_back(viewpoints.back()->getPosition());
        // }


    }

    csv_file.close();
    return 0;
}
