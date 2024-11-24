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
#include <queue>

#include <open3d/Open3D.h>

#include <mlpack/methods/kmeans/kmeans.hpp>

#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

#include <mlpack/core.hpp>
#include <armadillo>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkSphereSource.h>
#include <vtkIdTypeArray.h>


// GaussianParameters struct to hold the result
struct GaussianParameters {
    Eigen::Vector4d mean;
    Eigen::Matrix4d covariance;
    double weight;
};

// ClusterParameters struct to hold the result
struct ClusterParameters {
    double centroid;                     // Cluster center (potential value)
    std::vector<octomap::OcTreeKey> members; // Voxels belonging to this cluster
};



// struct PositionCluster {
//     Eigen::Vector3d centroid;
//     std::vector<octomap::OcTreeKey> members;
// };

struct Blob {
    Eigen::Vector3d centroid;
    double weightedSum; // Weighted sum of potentials
    std::vector<int> vertexIndices; // Indices of points in the blob
};

std::vector<ClusterParameters> fitKMeansToPotentials(
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    int num_clusters
) {
    // Step 1: Extract potential values
    std::vector<float> potentials;
    std::vector<octomap::OcTreeKey> voxelKeys;

    for (const auto& kv : voxelToSphereMap) {
        const auto& key = kv.first;

        // Get the potential value from the model
        float potential = boost::get<float>(model.getVoxelProperty(key, "potential"));

        potentials.push_back(potential);
        voxelKeys.push_back(key);
    }

    if (potentials.empty()) {
        throw std::runtime_error("No valid potential values found for clustering.");
    }

    // Step 2: Convert potentials to Armadillo matrix
    arma::mat data(1, potentials.size()); // 1 row (potential), N columns
    for (size_t i = 0; i < potentials.size(); ++i) {
        data(0, i) = potentials[i];
    }

    std::cout << "Data matrix size: " << data.n_rows << "x" << data.n_cols << std::endl;

    // Step 3: Apply KMeans
    arma::Row<size_t> assignments; // Cluster assignments for each potential
    arma::mat centroids;           // Centroids of the clusters

    mlpack::kmeans::KMeans<> kmeans;
    kmeans.Cluster(data, num_clusters, assignments, centroids);

    // Step 4: Extract cluster parameters
    std::vector<ClusterParameters> clusters(num_clusters);
    for (size_t i = 0; i < num_clusters; ++i) {
        clusters[i].centroid = centroids(0, i); // Centroid of potential values
    }

    // Step 5: Group voxels by cluster
    for (size_t i = 0; i < assignments.n_elem; ++i) {
        size_t cluster_id = assignments[i];
        clusters[cluster_id].members.push_back(voxelKeys[i]);
    }

    return clusters;
}

std::vector<PositionCluster> clusterHighPotentialVoxels(
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    const ClusterParameters& highPotentialCluster,
    int minClusterSize = 10 // Minimum size to remove noise
) {
    // Step 1: Filter high-potential voxels
    std::vector<Eigen::Vector3d> positions;
    std::vector<octomap::OcTreeKey> keys;

    for (const auto& kv : voxelToSphereMap) {
        const auto& key = kv.first;
        float potential = boost::get<float>(model.getVoxelProperty(key, "potential"));

        if (potential >= highPotentialCluster.centroid) {
            positions.push_back(kv.second); // Sphere position
            keys.push_back(key);
        }
    }

    if (positions.empty()) {
        throw std::runtime_error("No high-potential voxels found for clustering.");
    }

    // Step 2: Convert positions to Armadillo matrix
    arma::mat data(3, positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        data(0, i) = positions[i].x();
        data(1, i) = positions[i].y();
        data(2, i) = positions[i].z();
    }

    // Step 3: Apply DBSCAN
    arma::Row<size_t> assignments; // Cluster assignments for each voxel
    mlpack::dbscan::DBSCAN<> dbscan(50.0, minClusterSize); // Epsilon: 0.1 (adjust as needed)
    dbscan.Cluster(data, assignments);

    // Step 4: Extract clusters
    std::unordered_map<size_t, PositionCluster> clustersMap;
    for (size_t i = 0; i < assignments.n_elem; ++i) {
        size_t cluster_id = assignments[i];
        if (cluster_id == (size_t)-1) {
            continue; // Skip noise
        }

        if (clustersMap.find(cluster_id) == clustersMap.end()) {
            clustersMap[cluster_id] = PositionCluster{Eigen::Vector3d::Zero(), {}};
        }

        clustersMap[cluster_id].members.push_back(keys[i]);
        clustersMap[cluster_id].centroid += positions[i];
    }

    // Finalize centroids
    std::vector<PositionCluster> clusters;
    for (auto& kv : clustersMap) {
        auto& cluster = kv.second;
        cluster.centroid /= cluster.members.size();
        clusters.push_back(cluster);
    }

    return clusters;
}



#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <Eigen/Dense>


struct SphereBlob {
    Eigen::Vector3d centroid;
    double weightedPotential;
    std::vector<Eigen::Vector3d> points;
    Eigen::Vector3d highestPotentialVertex; // Vertex with the highest potential
    double highestPotentialValue = 0.0;    // Highest potential value
};



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


// Function to compute the mean and standard deviation of potential values
void computePotentialStatistics(
    const visioncraft::Model& model,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap,
    double& mean, double& std_dev)
{
     // Step 1: Extract potential values
    std::vector<float> potentials;
    for (const auto& kv : voxelToSphereMap) {
        const auto& key = kv.first;
        float potential = boost::get<float>(model.getVoxelProperty(key, "potential"));
        potentials.push_back(potential);
    }

    // Step 2: Calculate mean and standard deviation manually
    double sum = 0.0, sum_squared = 0.0;
    for (float potential : potentials) {
        sum += potential;
        sum_squared += potential * potential;
    }

    mean = sum / potentials.size();
    double variance = (sum_squared / potentials.size()) - (mean * mean);
    std_dev = std::sqrt(variance);

    // Step 3: Print out results
    std::cout << "Manual Calculation of Mean: " << mean << std::endl;
    std::cout << "Manual Calculation of Standard Deviation: " << std_dev << std::endl;

    // Step 4: Optional - Print all potentials
    std::cout << "Potentials: ";
    for (float potential : potentials) {
        std::cout << potential << " ";
    }
    std::cout << std::endl;

    // Step 5: Analyze distribution
    std::cout << "Number of Potentials: " << potentials.size() << std::endl;
    std::cout << "Max Potential: " << *std::max_element(potentials.begin(), potentials.end()) << std::endl;
    std::cout << "Min Potential: " << *std::min_element(potentials.begin(), potentials.end()) << std::endl;
}



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


std::vector<PositionCluster> getHighPotentialClusters(const visioncraft::Model& model,  std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap) {
    
    std::vector<PositionCluster> positionClusters;  // To store the resulting clusters
    
    try {
        int num_clusters = 2; // We assume 2 clusters for the first pass on potentials

        // Fit K-means to potentials
        auto clusters = fitKMeansToPotentials(model, voxelToSphereMap, num_clusters);

        // Select the cluster with the highest centroid value (assuming the centroid is a vector)
        const auto& highPotentialCluster = (clusters[0].centroid > clusters[1].centroid) ? clusters[0] : clusters[1];
        
        // Cluster high-potential voxels based on their position
        positionClusters = clusterHighPotentialVoxels(model, voxelToSphereMap, highPotentialCluster, 3);

  
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }

    return positionClusters;  // Return the position clusters
}


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
    // Normalize potentials
    // for (vtkIdType i = 0; i < interpolatedPotentials->GetNumberOfTuples(); ++i) {
    //     interpolatedPotentials->SetValue(i, interpolatedPotentials->GetValue(i) / maxPotential);
    // }

    spherePolyData->GetPointData()->AddArray(interpolatedPotentials);
    return spherePolyData;
}


// Function to compute interpolated potentials and extract blobs
std::vector<SphereBlob> computeInterpolatedBlobs(
    vtkSmartPointer<vtkPolyData> spherePolyData,
    float potential_threshold = 0.5,
    int min_blob_size = 10)
{
    // Step 1: Get potential values and validate data
    vtkSmartPointer<vtkFloatArray> potentials = vtkFloatArray::SafeDownCast(
        spherePolyData->GetPointData()->GetArray("potential"));
    if (!potentials) {
        throw std::runtime_error("Potentials data not found on sphere.");
    }

    vtkSmartPointer<vtkPoints> sphereVertices = spherePolyData->GetPoints();
    vtkSmartPointer<vtkCellArray> polys = spherePolyData->GetPolys();
    if (!sphereVertices || !polys) {
        throw std::runtime_error("Invalid vtkPolyData structure. Ensure points and polys are initialized.");
    }

    vtkIdType numVertices = sphereVertices->GetNumberOfPoints();

    // Step 2: Build adjacency list
    std::vector<std::vector<int>> adjacencyList(numVertices);
    polys->InitTraversal();
    vtkIdType npts;
    const vtkIdType* pts;
    while (polys->GetNextCell(npts, pts)) {
        for (vtkIdType i = 0; i < npts; ++i) {
            int v1 = pts[i];
            int v2 = pts[(i + 1) % npts];
            adjacencyList[v1].push_back(v2);
            adjacencyList[v2].push_back(v1);
        }
    }

    // Step 3: Identify high-potential vertices
    std::vector<bool> isHighPotential(numVertices, false);
    for (vtkIdType i = 0; i < numVertices; ++i) {
        if (potentials->GetValue(i) >= potential_threshold) {
            isHighPotential[i] = true;
        }
    }

    // Step 4: Perform clustering
    std::vector<SphereBlob> blobs;
    std::vector<bool> visited(numVertices, false);

    for (vtkIdType startIdx = 0; startIdx < numVertices; ++startIdx) {
        if (!isHighPotential[startIdx] || visited[startIdx]) continue;

        // New blob for the current cluster
        SphereBlob blob;
        std::queue<int> toVisit;
        toVisit.push(startIdx);

        Eigen::Vector3d sumPosition(0.0, 0.0, 0.0);
        double sumPotential = 0.0;
        size_t pointCount = 0;

        while (!toVisit.empty()) {
            int current = toVisit.front();
            toVisit.pop();

            if (visited[current]) continue;
            visited[current] = true;

            // Add current point to blob
            Eigen::Vector3d point(
                sphereVertices->GetPoint(current)[0],
                sphereVertices->GetPoint(current)[1],
                sphereVertices->GetPoint(current)[2]);
            blob.points.push_back(point);

            double currentPotential = potentials->GetValue(current);
            sumPosition += point;
            sumPotential += currentPotential;
            ++pointCount;

            if (currentPotential > blob.highestPotentialValue) {
                blob.highestPotentialValue = currentPotential;
                blob.highestPotentialVertex = point;
            }

            // Enqueue neighbors
            for (int neighbor : adjacencyList[current]) {
                if (isHighPotential[neighbor] && !visited[neighbor]) {
                    toVisit.push(neighbor);
                }
            }
        }

        // Skip blobs smaller than the minimum size
        if (pointCount >= static_cast<size_t>(min_blob_size)) {
            blob.centroid = sumPosition / pointCount;
            blob.weightedPotential = sumPotential / pointCount;
            blobs.push_back(blob);
        }
    }

    return blobs;
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
    int num_viewpoints = 5;
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

        // visualizer.visualizePotentialOnSphere(model, sphere_radius, "potential", voxelToSphereMap);
        
        // std::vector<PositionCluster> injection_regions = getHighPotentialClusters(model, voxelToSphereMap);
        // visualizer.visualizeInjectionRegionsOnSphere(model, injection_regions, voxelToSphereMap);

        float potential_threshold = 15.0f;
        int min_blob_size = 10;

        vtkSmartPointer<vtkPolyData> spherePolyData = computeInterpolatedPotentialsOnSphere(
            model, voxelToSphereMap, "potential", sphere_radius);

        // Step 3: Compute blobs based on interpolated potentials
        std::vector<SphereBlob> blobs = computeInterpolatedBlobs(
            spherePolyData, potential_threshold, min_blob_size);
        // Find the blob with the highest weight

        float MAX_POTENTIAL = std::log(M_PI * sphere_radius) * num_viewpoints;
        visualizer.visualizePotentialOnSphere(spherePolyData, MAX_POTENTIAL);

        // // Prepare a vector to store the centroids
        std::vector<Eigen::Vector3d> blobCentroids;

        // Extract the centroids from the blobs
        for (const auto& blob : blobs) {
            blobCentroids.push_back(blob.centroid); // Assuming each SphereBlob has a member `centroid`
        }

        // Visualize the blob centroids
        visualizer.visualizeBlobCentroidsOnSphere(model, blobCentroids, sphere_radius);

  
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


        // if (iter % 30 == 0 && iter != 0) { // Add a new viewpoint every 10 iterations
        //     // Eigen::Vector3d random_position(
        //     //     static_cast<float>(rand()) / RAND_MAX * sphere_radius,
        //     //     static_cast<float>(rand()) / RAND_MAX * sphere_radius,
        //     //     static_cast<float>(rand()) / RAND_MAX * sphere_radius);
        //     // Eigen::Vector3d look_at(0.0, 0.0, 0.0);

        //     // Add the new viewpoint
        //     addNewViewpoint(viewpoints, visibilityManager, visualizer, random_position, look_at, sphere_radius);

        //     // Initialize its previous position
        //     previous_positions.push_back(viewpoints.back()->getPosition());
        // }

        Eigen::Vector3d next_candidate;

        // // // Find the blob with the highest weight and its highest potential vertex if blobs exist
        if (!blobs.empty()) {
            const SphereBlob& highestWeightBlob = *std::max_element(
                blobs.begin(), blobs.end(),
                [](const SphereBlob& a, const SphereBlob& b) {
                    return a.weightedPotential < b.weightedPotential;
                });

            next_candidate = highestWeightBlob.highestPotentialVertex;
            std::cout << "Next candidate vertex: [" << next_candidate.x() << ", "
                    << next_candidate.y() << ", " << next_candidate.z() << "]" << std::endl;

            if (iter % 30 == 0 && iter != 0){
            
                Eigen::Vector3d look_at(0.0, 0.0, 0.0);
                // Add the new viewpoint
                addNewViewpoint(viewpoints, visibilityManager, visualizer, next_candidate, look_at, sphere_radius);

                // Initialize its previous position
                previous_positions.push_back(viewpoints.back()->getPosition());

            }
                
        }


    }

    csv_file.close();
    return 0;
}
