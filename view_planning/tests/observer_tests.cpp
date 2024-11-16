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

// Compute attractive force
Eigen::Vector3d computeAttractiveForce(
    const visioncraft::Model& model, 
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    float sphere_radius,
    float sigma, 
    int V_max) 
{
    Eigen::Vector3d F_attr = Eigen::Vector3d::Zero();
    const auto& voxelMap = model.getVoxelMap().getMap();

    for (const auto& kv : voxelMap) {
        const auto& voxel = kv.second;
        const auto& key = kv.first;

        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / V_max;

        Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();
        float distance_squared = r.squaredNorm();
        float W = std::exp(-(distance_squared - sphere_radius * sphere_radius) / (2 * sigma * sigma));
        Eigen::Vector3d grad_W = -r * W / (sigma * sigma);

        F_attr += (1.0f - V_norm) * grad_W;
    }
    return F_attr;
}

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
            float distance = r.norm() + 1e-5f;
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

double computeAttractivePotentialEnergy(
    const visioncraft::Model& model,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    float sphere_radius,
    float sigma,
    int V_max) {
    double U_attr = 0.0;
    const auto& voxelMap = model.getVoxelMap().getMap();

    for (const auto& kv : voxelMap) {
        const auto& voxel = kv.second;
        const auto& key = kv.first;

        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / V_max;

        for (const auto& viewpoint : viewpoints) {
            Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();
            double distance_squared = r.squaredNorm();
            double W = std::exp(-(distance_squared - sphere_radius * sphere_radius) / (2 * sigma * sigma));

            // Accumulate the potential energy
            U_attr += (1.0 - V_norm) * W;
        }
    }

    return U_attr;
}


// Compute total system energy
double computeSystemEnergy(
    const visioncraft::Model& model,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints,
    float sigma,
    float k_repel,
    float alpha) {
    double U_attr = 0.0;
    double U_repel = 0.0;

    // Attractive potential energy
    const auto& voxelMap = model.getVoxelMap().getMap();
    for (const auto& kv : voxelMap) {
        const auto& voxel = kv.second;
        const auto& key = kv.first;

        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / viewpoints.size();

        double closest_contribution = std::numeric_limits<double>::max();
        for (const auto& viewpoint : viewpoints) {
            Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();
            double distance_squared = r.squaredNorm();
            double W = std::exp(-distance_squared / (2 * sigma * sigma));
            closest_contribution = std::min(closest_contribution, (1.0 - V_norm) * W);
        }
        U_attr += closest_contribution;
    }

    // Repulsive potential energy
    for (size_t i = 0; i < viewpoints.size(); ++i) {
        for (size_t j = i + 1; j < viewpoints.size(); ++j) {
            Eigen::Vector3d r = viewpoints[i]->getPosition() - viewpoints[j]->getPosition();
            double distance = r.norm() + 1e-5;  // Avoid division by zero
            double repulsion = k_repel / std::pow(distance, alpha);
            U_repel += repulsion;
        }
    }

    return U_attr + U_repel;
}


double computeAverageForceMagnitude(const visioncraft::Model& model, const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints, float sphere_radius, float sigma, float k_repel, float alpha) {
    double total_force_magnitude = 0.0;

    for (const auto& viewpoint : viewpoints) {
        Eigen::Vector3d F_attr = computeAttractiveForce(model, viewpoint, sphere_radius, sigma, viewpoints.size());
        Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, k_repel, alpha);
        Eigen::Vector3d F_total = F_attr + F_repel;
        total_force_magnitude += F_total.norm();
    }

    return total_force_magnitude / viewpoints.size();
}

double computeAverageViewpointMovement(const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& previous_positions, const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& current_positions) {
    double total_movement = 0.0;

    for (size_t i = 0; i < previous_positions.size(); ++i) {
        Eigen::Vector3d prev_pos = previous_positions[i]->getPosition();
        Eigen::Vector3d curr_pos = current_positions[i]->getPosition();
        total_movement += (curr_pos - prev_pos).norm();
    }

    return total_movement / previous_positions.size();
}

double computeCoverageScoreChange(double current_score, double previous_score) {
    return std::abs(current_score - previous_score);
}

#include <vector>
#include <memory>
#include <Eigen/Dense>

// Compute kinetic energy of the system
double computeKineticEnergy(
    const std::vector<Eigen::Vector3d>& previous_positions,
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& current_viewpoints,
    double delta_t) {
    double kinetic_energy = 0.0;

    for (size_t i = 0; i < previous_positions.size(); ++i) {
        Eigen::Vector3d prev_pos = previous_positions[i];
        Eigen::Vector3d curr_pos = current_viewpoints[i]->getPosition();

        // Approximate velocity
        Eigen::Vector3d velocity = (curr_pos - prev_pos) / delta_t;

        // Compute kinetic energy contribution (mass assumed 1.0)
        kinetic_energy += 0.5 * velocity.squaredNorm();
    }

    return kinetic_energy;
}


// double computeVisibilityHomogeneity(const visioncraft::Model& model) {
//     const auto& voxelMap = model.getVoxelMap().getMap();
//     std::vector<double> visibility_values;

//     // Collect visibility values
//     for (const auto& kv : voxelMap) {
//         const auto& key = kv.first;
//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         visibility_values.push_back(static_cast<double>(visibility));
//     }

//     if (visibility_values.empty()) {
//         return 0.0; // No voxels, return 0 as default
//     }

//     // Compute mean
//     double sum = std::accumulate(visibility_values.begin(), visibility_values.end(), 0.0);
//     double mean = sum / visibility_values.size();

//     // Compute standard deviation
//     double variance = 0.0;
//     for (double value : visibility_values) {
//         variance += std::pow(value - mean, 2);
//     }
//     variance /= visibility_values.size();
//     double std_dev = std::sqrt(variance);

//     // Coefficient of Variation (CV): std_dev / mean
//     return std_dev / mean;
// }

double computeVisibilityHomogeneity(const visioncraft::Model& model) {
    const auto& voxelMap = model.getVoxelMap().getMap();
    std::vector<int> visibility_values;

    // Collect the visibility values
    double total_visibility = 0.0;
    for (const auto& kv : voxelMap) {
        int visibility = boost::get<int>(model.getVoxelProperty(kv.first, "visibility"));
        visibility_values.push_back(visibility);
        total_visibility += visibility;
    }

    if (total_visibility == 0 || visibility_values.empty()) {
        return 0.0; // No visibility, assume zero Gini
    }

    // Step 1: Compute the frequency of each visibility value
    std::unordered_map<int, int> visibility_counts;
    for (int visibility : visibility_values) {
        visibility_counts[visibility]++;
    }

    // Step 2: Compute probabilities for each visibility value
    double gini = 1.0;
    for (const auto& count : visibility_counts) {
        double p = static_cast<double>(count.second) / visibility_values.size(); // Probability of the visibility value
        gini -= p * p; // Add the squared probability for the Gini index
    }

    return gini;
}


double computeCoefficientOfVariation(const visioncraft::Model& model) {
    const auto& voxelMap = model.getVoxelMap().getMap();
    std::vector<int> visibility_values;

    for (const auto& kv : voxelMap) {
        int visibility = boost::get<int>(model.getVoxelProperty(kv.first, "visibility"));
        visibility_values.push_back(visibility);
    }

    if (visibility_values.empty()) {
        return 0.0; // No voxels, assume perfect uniformity
    }

    double mean = std::accumulate(visibility_values.begin(), visibility_values.end(), 0.0) / visibility_values.size();
    double variance = 0.0;

    for (int value : visibility_values) {
        variance += (value - mean) * (value - mean);
    }

    double stddev = std::sqrt(variance / visibility_values.size());
    return stddev / mean; // Coefficient of Variation
}

double computeVisibilityEntropy(const visioncraft::Model& model) {
    const auto& voxelMap = model.getVoxelMap().getMap();
    std::vector<int> visibility_values;

    double total_visibility = 0.0;
    for (const auto& kv : voxelMap) {
        int visibility = boost::get<int>(model.getVoxelProperty(kv.first, "visibility"));
        visibility_values.push_back(visibility);
        total_visibility += visibility;
    }

    if (total_visibility == 0 || visibility_values.empty()) {
        return 0.0; // No visibility, assume zero entropy
    }

    double entropy = 0.0;
    for (int visibility : visibility_values) {
        double p = visibility / total_visibility;
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
    }

    return entropy;
}

double computeHerfindahlIndex(const visioncraft::Model& model) {
    const auto& voxelMap = model.getVoxelMap().getMap();
    std::vector<int> visibility_values;

    double total_visibility = 0.0;
    for (const auto& kv : voxelMap) {
        int visibility = boost::get<int>(model.getVoxelProperty(kv.first, "visibility"));
        visibility_values.push_back(visibility);
        total_visibility += visibility;
    }

    if (total_visibility == 0 || visibility_values.empty()) {
        return 1.0; // All zero visibility, assume maximum concentration
    }

    double hhi = 0.0;
    for (int visibility : visibility_values) {
        double p = visibility / total_visibility;
        hhi += p * p;
    }

    return hhi;
}


int main() {
    srand(time(nullptr));

    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    visioncraft::Model model;
    std::cout << "Loading model..." << std::endl;
    model.loadModel("../models/cube.ply", 100000);
    std::cout << "Model loaded successfully." << std::endl;

    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);

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
    float sigma = 50.0f;
    float k_repel = 15000.0f;
    float delta_t = 0.2f;
    float alpha = 1.0f;
    int max_iterations = 200;

    // Prepare CSV logging
    std::ofstream csv_file("results.csv");
    csv_file << "Timestep,CoverageScore,SystemEnergy,KineticEnergy,VisibilityHomogeneity,VisibilityEntropy,CoefficientOfVariation,ForceMagnitude,ViewpointMovement,CoverageChange\n";
    csv_file << std::fixed << std::setprecision(6);

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

        // Compute metrics
        double coverage_score = visibilityManager->computeCoverageScore();
        double system_energy = computeSystemEnergy(model, viewpoints, sigma, k_repel, alpha);
        double attractive_potential_energy = computeAttractivePotentialEnergy(model, viewpoints, sphere_radius, sigma, num_viewpoints);
        double kinetic_energy = computeKineticEnergy(previous_positions, viewpoints, delta_t);
        double visibility_homogeneity = computeVisibilityHomogeneity(model); // Gini coefficient
        double visibility_entropy = computeVisibilityEntropy(model);         // Entropy
        double coefficient_of_variation = computeCoefficientOfVariation(model); // Coefficient of Variation
       
        // Print metrics
        std::cout << "Iteration: " << iter << ", "
                << "Coverage Score: " << coverage_score << ", "
                << "System Energy: " << system_energy << ", "
                << "Attractive Potential Energy: " << attractive_potential_energy << ", "
                << "Kinetic Energy: " << kinetic_energy << ", "
                << "Visibility Homogeneity: " << visibility_homogeneity << ", "
                << "Visibility Entropy: " << visibility_entropy << ", "
                << "Coefficient of Variation: " << coefficient_of_variation << "\n";

        // Log metrics to CSV
        csv_file << iter << "," << coverage_score << "," << system_energy << ","
                << attractive_potential_energy << "," << kinetic_energy << ","
                << visibility_homogeneity << "," << visibility_entropy << ","
                << coefficient_of_variation << "\n";

         for (size_t i = 0; i < viewpoints.size(); ++i) {
            // Update previous positions
            previous_positions[i] = viewpoints[i]->getPosition();
        }
        // Update viewpoint positions
        for (auto& viewpoint : viewpoints) {
         

            Eigen::Vector3d F_attr = computeAttractiveForce(model, viewpoint, sphere_radius, sigma, num_viewpoints);
            Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, k_repel, alpha);

            Eigen::Vector3d F_total = F_attr + F_repel;
            Eigen::Vector3d n = viewpoint->getPosition().normalized();
            Eigen::Vector3d F_tangent = F_total - F_total.dot(n) * n;
            Eigen::Vector3d new_position = viewpoint->getPosition() + delta_t * F_tangent;
            updateViewpointState(viewpoint, new_position, sphere_radius);

            visualizer.addViewpoint(*viewpoint, false, true);
        }

        visualizer.addVoxelMapProperty(model, "visibility");
        visualizer.render();
        visualizer.removeViewpoints();
        visualizer.removeVoxelMapProperty();

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    csv_file.close();
    return 0;
}
