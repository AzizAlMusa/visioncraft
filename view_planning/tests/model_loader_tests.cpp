#include "visioncraft/model_loader.h" // Include your ModelLoader header
#include <octomap/octomap.h>
#include <open3d/Open3D.h>

// Helper function to visualize a point cloud
void visualizePointCloud(const std::shared_ptr<open3d::geometry::PointCloud>& pointCloud, const std::string& title) {
    if (pointCloud && !pointCloud->points_.empty()) {
        open3d::visualization::DrawGeometries({pointCloud}, title, 1600, 900, 50, 50, true);
    } else {
        std::cerr << "No point cloud data to visualize." << std::endl;
    }
}

// Helper function to visualize multiple point clouds with toggle visibility
void visualizePointCloudsWithToggle(const std::shared_ptr<open3d::geometry::PointCloud>& volumetricPointCloud, 
                                    const std::shared_ptr<open3d::geometry::PointCloud>& surfaceShellPointCloud, 
                                    const std::string& title) {
    open3d::visualization::VisualizerWithKeyCallback visualizer;
    visualizer.CreateVisualizerWindow(title, 1600, 900);

    visualizer.AddGeometry(volumetricPointCloud);
    visualizer.AddGeometry(surfaceShellPointCloud);

    bool surfaceShellVisible = true;
    visualizer.RegisterKeyCallback(GLFW_KEY_T, [&](open3d::visualization::Visualizer* vis) {
        surfaceShellVisible = !surfaceShellVisible;
        surfaceShellPointCloud->Clear();
        if (surfaceShellVisible) {
            // Re-add the points to the surfaceShellPointCloud
            for (const auto& point : surfaceShellPointCloud->points_) {
                surfaceShellPointCloud->points_.push_back(point);
            }
            for (const auto& color : surfaceShellPointCloud->colors_) {
                surfaceShellPointCloud->colors_.push_back(color);
            }
        }
        vis->UpdateGeometry();
        return true;
    });

    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
}

// Helper function to visualize a voxel grid
void visualizeVoxelGrid(const std::shared_ptr<open3d::geometry::VoxelGrid>& voxelGrid, const std::string& title) {
    if (voxelGrid && voxelGrid->voxels_.size() > 0) {
        open3d::visualization::Visualizer visualizer;
        visualizer.CreateVisualizerWindow(title, 1600, 900);
        visualizer.AddGeometry(voxelGrid);
        visualizer.GetRenderOption().ToggleLightOn();
        visualizer.GetRenderOption().mesh_shade_option_ = open3d::visualization::RenderOption::MeshShadeOption::FlatShade;
        visualizer.Run();
        visualizer.DestroyVisualizerWindow();
    } else {
        std::cerr << "No voxel grid data to visualize." << std::endl;
    }
}

// Function to convert an OctoMap to an Open3D point cloud
std::shared_ptr<open3d::geometry::PointCloud> ConvertOctomapToPointCloud(const std::shared_ptr<octomap::ColorOcTree>& octomap, const open3d::utility::optional<Eigen::Vector3d>& color = {}) {
    auto pointCloud = std::make_shared<open3d::geometry::PointCloud>();
    for (auto it = octomap->begin_leafs(); it != octomap->end_leafs(); ++it) {
        if (it->getOccupancy() > 0.5) {
            pointCloud->points_.emplace_back(it.getX(), it.getY(), it.getZ());
            if (color.has_value()) {
                pointCloud->colors_.emplace_back(color.value());
            } else {
                pointCloud->colors_.emplace_back(it->getColor().r / 255.0, it->getColor().g / 255.0, it->getColor().b / 255.0);
            }
        }
    }
    return pointCloud;
}

// Function to load the mesh
bool loadMesh(visioncraft::ModelLoader& modelLoader, const std::string& filePath) {
    if (!modelLoader.loadMesh(filePath)) {
        std::cerr << "Failed to load mesh." << std::endl;
        return false;
    }
    std::cout << "Mesh loaded successfully." << std::endl;
    return true;
}

// Function to generate the point cloud
bool generatePointCloud(visioncraft::ModelLoader& modelLoader) {
    if (!modelLoader.generatePointCloud(10000)) {
        std::cerr << "Failed to generate point cloud." << std::endl;
        return false;
    }
    std::cout << "Point cloud generated successfully." << std::endl;
    return true;
}

// Function to generate the voxel grid
bool generateVoxelGrid(visioncraft::ModelLoader& modelLoader) {
    if (!modelLoader.generateVoxelGrid(0.0)) {
        std::cerr << "Failed to generate voxel grid." << std::endl;
        return false;
    }
    std::cout << "Voxel grid generated successfully." << std::endl;
    return true;
}

// Function to generate the exploration map
bool generateExplorationMap(visioncraft::ModelLoader& modelLoader) {
    octomap::point3d min_bound = modelLoader.getMinBound();
    octomap::point3d max_bound = modelLoader.getMaxBound();
    if (!modelLoader.generateExplorationMap(32, min_bound, max_bound)) {
        std::cerr << "Failed to generate exploration map." << std::endl;
        return false;
    }
    std::cout << "Exploration map generated successfully." << std::endl;
    return true;
}

// Function to generate the octomap
bool generateOctomap(visioncraft::ModelLoader& modelLoader) {
    double resolution = modelLoader.getExplorationMapResolution();
    if (!modelLoader.generateOctoMap(resolution)) {
        std::cerr << "Failed to generate octomap." << std::endl;
        return false;
    }
    std::cout << "Octomap generated successfully." << std::endl;
    return true;
}

// Function to generate the volumetric octomap
bool generateVolumetricOctomap(visioncraft::ModelLoader& modelLoader) {
    double resolution = modelLoader.getAverageSpacing();
    if (!modelLoader.generateVolumetricOctoMap(resolution)) {
        std::cerr << "Failed to generate volumetric octomap." << std::endl;
        return false;
    }
    std::cout << "Volumetric octomap generated successfully." << std::endl;
    return true;
}

// Function to generate the surface shell octomap
bool generateSurfaceShellOctomap(visioncraft::ModelLoader& modelLoader) {
    if (!modelLoader.generateSurfaceShellOctomap()) {
        std::cerr << "Failed to generate surface shell octomap." << std::endl;
        return false;
    }
    std::cout << "Surface shell octomap generated successfully." << std::endl;
    return true;
}

int main(int argc, char** argv) {
    // Check if the file path argument is provided
    if (argc != 2) {
        std::cerr << "Usage: model_loader_tests <file_path>" << std::endl;
        return 1;
    }

    // Get the file path from the command-line argument
    std::string filePath = argv[1];
    std::cout << "File path: " << filePath << std::endl;

    // Create an instance of the ModelLoader
    visioncraft::ModelLoader modelLoader;

    // if (modelLoader.loadModel(filePath, 10000, -1)) {
    //     std::cout << "Model loaded successfully." << std::endl;
    // } else {
    //     std::cerr << "Failed to load model." << std::endl;
    //     return 1;
    // }

    
    // Load the mesh
    if (!loadMesh(modelLoader, filePath)) return 1;

    // Initialize the raycasting scene
    if (!modelLoader.initializeRaycastingScene()) return 1;

    // Generate the point cloud
    int num_samples = 10000;
    if (!modelLoader.generatePointCloud(num_samples)) return 1;

    // Generate Volumetric Point Cloud
    if (!modelLoader.generateVolumetricPointCloud()) return 1;

    // Generate the voxel grid
    double resolution = modelLoader.getAverageSpacing();
    if (!modelLoader.generateVoxelGrid(resolution)) return 1;

    // Generate the octomap
    if (!modelLoader.generateOctoMap(resolution)) return 1;

    // Generate the volumetric octomap
    if (!modelLoader.generateVolumetricOctoMap(resolution)) return 1;

    // Generate the surface shell octomap
    if (!modelLoader.generateSurfaceShellOctomap()) return 1;

    // Generate the exploration map
    if (!modelLoader.generateExplorationMap(resolution, modelLoader.getMinBound(), modelLoader.getMaxBound())) return 1;

    return 0;
}
