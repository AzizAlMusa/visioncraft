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

// Function to map voxels to sphere positions
// Function to map voxels to sphere positions
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
                Eigen::Vector3d nearestSpherePoint = unoccludedPositions[nearestIdx];
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



// Compute potential for each voxel based on distances to all viewpoints
void computeVoxelPotentials(
    visioncraft::Model& model,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    int V_max,
    bool use_exponential,
    float sigma = 1.0f) // Default value; adjust as needed
{
    const auto& voxelMap = model.getVoxelMap().getMap();

    // Loop over each voxel in the map
    for (const auto& kv : voxelMap) {
        const auto& voxel = kv.second;
        const auto& key = kv.first;
        
        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / V_max;

        // V_norm > 1.0f ? V_norm = 1.0f : V_norm;

        float potential = 0.0f;
        for (const auto& viewpoint : viewpoints) {
            Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition() ;
            float distance_squared = r.squaredNorm();

            if (use_exponential) {
                potential += std::exp(-distance_squared / (2.0f * sigma * sigma));
                model.setVoxelProperty(key, "potential", potential);
            } else {
                potential += distance_squared;
            }
        } 
        model.setVoxelProperty(key, "potential",  (1.0f - V_norm) * potential);
        // std::cout << "Potential: " << (1.0f - V_norm) * potential << std::endl;
    }


}

Eigen::Vector3d computeAttractiveForce(
    const visioncraft::Model& model,
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    float sigma,
    int V_max)
{   

    Eigen::Vector3d F_attr = Eigen::Vector3d::Zero();
    const auto& voxelMap = model.getVoxelMap().getMap();

    for (const auto& kv : voxelMap) {
        const auto& voxel = kv.second;
        const auto& key = kv.first;

        // Check if voxel potential was computed
        float potential = boost::get<float>(model.getVoxelProperty(key, "potential"));
        


        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / V_max;

        // V_norm > 1.0f ? V_norm = 1.0f : V_norm;

        Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();

        // Gradient of the potential with respect to the viewpoint position
        Eigen::Vector3d grad_U = 2 * (1.0f - V_norm) * r ;

        F_attr -= grad_U / sigma;
    
    }

    return F_attr;
}

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





// Compute repulsive force
Eigen::Vector3d computeRepulsiveForce(
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints, 
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint, 
    float k_repel, 
    float alpha) 
{
    Eigen::Vector3d F_repel = Eigen::Vector3d::Zero();

    for (const auto& other_viewpoint : viewpoints) {
        if (viewpoint != other_viewpoint) {
            Eigen::Vector3d r = viewpoint->getPosition() - other_viewpoint->getPosition();
            double distance = r.norm() + 1e-5;
            Eigen::Vector3d force = k_repel * r / std::pow(distance, alpha + 1.0f);
            F_repel += force;
        }
    }
    return F_repel;
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
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    float k_repel,
    float alpha,
    float sigma,
    int V_max)
{
    double total_force_magnitude = 0.0;

    for (const auto& viewpoint : viewpoints) {
        Eigen::Vector3d F_attr = computeAttractiveForce(model, viewpoint, sigma, V_max);
        Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, k_repel, alpha);
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
    model.loadModel("../models/gorilla.ply", 100000);
    std::cout << "Model loaded successfully." << std::endl;

    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);
    model.addVoxelProperty("potential", 0.0f);

    float sphere_radius = 400.0f;
    int num_viewpoints = 2;
    auto viewpoints = generateClusteredViewpoints(num_viewpoints, sphere_radius);

    for (auto& viewpoint : viewpoints) {
        viewpoint->setDownsampleFactor(8.0);
        visibilityManager->trackViewpoint(viewpoint);
        viewpoint->setFarPlane(900);
        viewpoint->setNearPlane(300);
        
    }

    // Simulation parameters
    float sigma = 100.0f;
    float k_repel = 15000.0f;
    float delta_t = 0.2f;
    float alpha = 1.0f;
    int max_iterations = 100;
    int V_max = num_viewpoints; //num_viewpoints

    // Generate manifold mapping
    std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash> voxelToSphereMap;
    mapVoxelsToSphere(model, sphere_radius, voxelToSphereMap);

    // print the sphere points
    for (const auto& kv : voxelToSphereMap) {
        std::cout << " sphere point: " << kv.second.transpose() << std::endl;
    }
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


    for (int iter = 0; iter < max_iterations; ++iter) {

        // Perform raycasting
        for (auto& viewpoint : viewpoints) {
            viewpoint->performRaycastingOnGPU(model);
        }
        visualizer.visualizePotentialOnSphere(model, sphere_radius, "potential", voxelToSphereMap);
        // Log viewpoint positions to the CSV file
        for (size_t i = 0; i < viewpoints.size(); ++i) {
            Eigen::Vector3d position = viewpoints[i]->getPosition();
            viewpoint_csv_file << iter << "," << i << ","
                            << position.x() << "," << position.y() << "," << position.z() << "\n";
        }

        bool use_exponential = false; // Set to false for quadratic potential

        computeVoxelPotentials(model, viewpoints, V_max, use_exponential, sigma);

        // Compute metrics
        double coverage_score = visibilityManager->computeCoverageScore();
  
       // Compute system energy
        double system_energy = computeSystemEnergy(model, viewpoints, sigma, V_max, k_repel, alpha);
        double kinetic_energy = computeKineticEnergy(viewpoints, previous_positions, delta_t);
        double average_force = computeAverageForce(model, viewpoints, k_repel, alpha, sigma, V_max);
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
            Eigen::Vector3d F_attr = computeAttractiveForce(model, viewpoint, sigma, V_max);
            Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, k_repel, alpha);
            std::cout << "F_attr: " << F_attr.transpose() << "F_repel: " << F_repel.transpose() << std::endl;
 

            Eigen::Vector3d F_total = F_attr + F_repel ; // + F_repel
            Eigen::Vector3d n = viewpoint->getPosition().normalized();
            Eigen::Vector3d F_tangent = F_total - F_total.dot(n) * n;
            // compute the percentage of f_tanget over the total force
            double percentage = F_tangent.norm() / F_total.norm();
            std::cout << "Percentage: " << percentage << std::endl;

            std::cout << "F_tangent: " << F_tangent.transpose() << std::endl;
            Eigen::Vector3d new_position = viewpoint->getPosition() + delta_t * F_tangent;
            updateViewpointState(viewpoint, new_position, sphere_radius);

            // visualizer.addViewpoint(*viewpoint, false, true);
            visualizer.updateViewpoint(*viewpoint, false, true, true, true);
        }

        visualizer.addVoxelMapProperty(model, "visibility");
        visualizer.render();
        // visualizer.removeViewpoints();
        visualizer.removeVoxelMapProperty();

        std::this_thread::sleep_for(std::chrono::milliseconds(1));


        // if (iter % 1 == 0 && iter != 0 && iter <=20) { // Add a new viewpoint every 10 iterations
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
