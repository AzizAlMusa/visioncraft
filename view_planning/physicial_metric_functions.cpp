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


// double computeAverageForceMagnitude(const visioncraft::Model& model, const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints, float sphere_radius, float sigma, float k_repel, float alpha) {
//     double total_force_magnitude = 0.0;

//     for (const auto& viewpoint : viewpoints) {
//         Eigen::Vector3d F_attr = computeAttractiveForce(model, viewpoint, sphere_radius, sigma, viewpoints.size());
//         Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, k_repel, alpha);
//         Eigen::Vector3d F_total = F_attr + F_repel;
//         total_force_magnitude += F_total.norm();
//     }

//     return total_force_magnitude / viewpoints.size();
// }

double computeAverageViewpointMovement(const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& previous_positions, const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& current_positions) {
    double total_movement = 0.0;

    for (size_t i = 0; i < previous_positions.size(); ++i) {
        Eigen::Vector3d prev_pos = previous_positions[i]->getPosition();
        Eigen::Vector3d curr_pos = current_positions[i]->getPosition();
        total_movement += (curr_pos - prev_pos).norm();
    }

    return total_movement / previous_positions.size();
}


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


// Compute attractive force
// Eigen::Vector3d computeAttractiveForce(
//     const visioncraft::Model& model, 
//     const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
//     float sphere_radius,
//     float sigma, 
//     int V_max) 
// {
//     Eigen::Vector3d F_attr = Eigen::Vector3d::Zero();
//     const auto& voxelMap = model.getVoxelMap().getMap();

//     for (const auto& kv : voxelMap) {
//         const auto& voxel = kv.second;
//         const auto& key = kv.first;

//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         float V_norm = static_cast<float>(visibility) / V_max;

//         Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();
//         float distance_squared = r.squaredNorm();
//         float W = std::exp(-(distance_squared - sphere_radius * sphere_radius) / (2 * sigma * sigma));
//         // float W = std::exp(-(distance_squared) / (2 * sigma * sigma));
        
//         Eigen::Vector3d grad_W = -r * W / (sigma * sigma);
        
//         F_attr += (1.0f - V_norm) * grad_W;

//         std::cout << "r norm: " << r.norm() << std::endl;
//         std::cout << "distance_squared: " << distance_squared << std::endl;
//         std::cout << "W: " << W << std::endl;
//         std::cout << "grad_W: " << grad_W << std::endl;
//         std::cout << "F_attr: " << F_attr << std::endl;

//     }
//     return F_attr;
// }

// Eigen::Vector3d computeAttractiveForce(
//     const visioncraft::Model& model, 
//     const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
//     float sphere_radius,
//     float sigma, 
//     int V_max) 
// {
//     Eigen::Vector3d F_attr = Eigen::Vector3d::Zero();
//     const auto& voxelMap = model.getVoxelMap().getMap();

//     // Precompute D^2 for efficiency, this is the normalization constant squared
//     float D_squared = sigma;

//     for (const auto& kv : voxelMap) {
//         const auto& voxel = kv.second;
//         const auto& key = kv.first;

//         // Get voxel visibility
//         int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
//         float V_norm = static_cast<float>(visibility) / V_max;

//         // Only consider under-observed voxels
//         if (V_norm < 1.0f) {
//             // Compute the distance vector (r) from the viewpoint to the voxel
//             Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();
//             float r_squared = r.squaredNorm();  // r^2

//             // Compute the weight function W(r)
//             float W = r_squared / D_squared;  // W(r) = r^2 / D^2

//             // Compute the gradient of W(r)
//             Eigen::Vector3d grad_W = 2.0f * r / D_squared;  // grad_W(r) = 2 * r / D^2

//             // Compute the attractive force contribution for this voxel
//             F_attr -= (1.0f - V_norm) * grad_W;
//         }
//     }

//     return F_attr;
// }

double system_energy = computeSystemEnergy(model, viewpoints, sigma, k_repel, alpha);
double attractive_potential_energy = computeAttractivePotentialEnergy(model, viewpoints, sphere_radius, sigma, num_viewpoints);
double kinetic_energy = computeKineticEnergy(previous_positions, viewpoints, delta_t);
double visibility_homogeneity = computeVisibilityHomogeneity(model); // Gini coefficient
double visibility_entropy = computeVisibilityEntropy(model);         // Entropy
double coefficient_of_variation = computeCoefficientOfVariation(model); // Coefficient of Variation


        // Print metrics
        std::cout << "Iteration: " << iter << ", "
                << "Coverage Score: " << coverage_score << ", " <<"\n";
        //         << "System Energy: " << system_energy << ", "
        //         << "Attractive Potential Energy: " << attractive_potential_energy << ", "
        //         << "Kinetic Energy: " << kinetic_energy << ", "
        //         << "Visibility Homogeneity: " << visibility_homogeneity << ", "
        //         << "Visibility Entropy: " << visibility_entropy << ", "
        //         << "Coefficient of Variation: " << coefficient_of_variation << "\n";