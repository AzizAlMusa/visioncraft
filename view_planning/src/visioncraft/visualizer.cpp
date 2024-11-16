#include "visioncraft/visualizer.h"
#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCamera.h>
#include <vtkFrustumSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkOctreePointLocator.h>
#include <vtkPoints.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPointData.h>
#include <vtkColorTransferFunction.h>
#include <vtkPLYReader.h>
#include <vtkSTLReader.h>
#include <vtkCubeSource.h>
#include <vtkAppendPolyData.h>
#include <vtkLineSource.h>
#include <vtkTransform.h> 
#include <vtkMatrix4x4.h>
#include <vtkMatrix3x3.h>
#include <vtkLine.h>  
#include <vtkVoxel.h>
#include <vtkGlyph3D.h>


#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkInteractorStyleTrackballActor.h>

#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

#include <iterator> // Provides std::begin and std::end

namespace visioncraft {


Visualizer::Visualizer() {
    renderer = vtkSmartPointer<vtkRenderer>::New();
    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

    renderWindow->SetSize(1920, 1080);
    renderWindow->AddRenderer(renderer);
    renderWindowInteractor->SetRenderWindow(renderWindow);
    renderWindow->SetWindowName("Visualization");

     vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    renderWindowInteractor->SetInteractorStyle(style);
}

Visualizer::~Visualizer() {}

void Visualizer::initializeWindow(const std::string& windowName) {
    // std::cout << "INITIALIZE WINDOW CALLED!!!!!" << std::endl;
    // renderWindow->SetSize(1280, 720);
    // renderWindow->AddRenderer(renderer);
    // renderWindowInteractor->SetRenderWindow(renderWindow);
    // renderWindow->SetWindowName(windowName.c_str());

    //  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    // renderWindowInteractor->SetInteractorStyle(style);

 
}

void visioncraft::Visualizer::processEvents() {
    if (renderWindowInteractor) {
        renderWindowInteractor->ProcessEvents();
    }
}

void visioncraft::Visualizer::startAsyncRendering() {
    stopRendering_ = false;

    // Start the rendering thread
    renderThread_ = std::thread([this]() {
        renderWindowInteractor->Initialize();

        // Run until stopRendering_ is set to true
        while (!stopRendering_) {
            renderWindow->Render();            // Render the current scene
            renderWindowInteractor->ProcessEvents();  // Process user interaction
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // Limit to ~60 FPS
        }
    });
}


void visioncraft::Visualizer::stopAsyncRendering() {
    stopRendering_ = true;
    if (renderThread_.joinable()) {
        renderThread_.join();
    }
}


void visioncraft::Visualizer::addViewpoint(const visioncraft::Viewpoint& viewpoint, bool showFrustum, bool showAxes) {
    std::vector<vtkSmartPointer<vtkActor>> actors;

    if (showAxes) {
        auto axesActors = this->showAxes(viewpoint.getPosition(), viewpoint.getOrientationMatrix());
        actors.insert(actors.end(), axesActors.begin(), axesActors.end());
    }

    if (showFrustum) {
        auto frustumActors = this->showFrustum(viewpoint);
        actors.insert(actors.end(), frustumActors.begin(), frustumActors.end());
    }

    for (auto& actor : actors) {
        renderer->AddActor(actor);  // Add each actor to the renderer
        viewpointActorMap_[viewpoint.getId()].push_back(actor);  // Add actor to the map
    }
}



void visioncraft::Visualizer::updateViewpoint(const visioncraft::Viewpoint& viewpoint, bool updateFrustum, bool updateAxes) {
    // Remove the existing viewpoint
    removeViewpoint(viewpoint);

    // Add the updated viewpoint
    addViewpoint(viewpoint, updateFrustum, updateAxes);
}




void Visualizer::addMultipleViewpoints(const std::vector<visioncraft::Viewpoint>& viewpoints) {
    for (const auto& viewpoint : viewpoints) {
        addViewpoint(viewpoint);
    }
}

void visioncraft::Visualizer::removeViewpoint(const visioncraft::Viewpoint& viewpoint) {
    auto it = viewpointActorMap_.find(viewpoint.getId());
    if (it != viewpointActorMap_.end()) {
        for (auto& actor : it->second) {
            renderer->RemoveActor(actor);  // Remove each actor individually
        }
        viewpointActorMap_.erase(it);  // Remove the entry from the map
    }
}




void Visualizer::removeViewpoints() {
    for (auto& actor : viewpointActors_) {
        renderer->RemoveActor(actor);  // Remove each actor from the renderer
    }
    viewpointActors_.clear();  // Clear the list of actors
}


void Visualizer::showRays(visioncraft::Viewpoint& viewpoint, const Eigen::Vector3d& color) {
    // Get the rays generated from the viewpoint
    auto rays = viewpoint.generateRays();

    // Get the viewpoint position
    Eigen::Vector3d viewpointPosition = viewpoint.getPosition();

    // Create a vtkPoints object to store all the points of the rays
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    // Create a vtkCellArray to store the line connectivity
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();

    // Iterate through the rays and add each line (ray) to the points and lines arrays
    for (const auto& rayEnd : rays) {
        // Insert the start and end points of the ray
        vtkIdType startId = points->InsertNextPoint(viewpointPosition(0), viewpointPosition(1), viewpointPosition(2));
        vtkIdType endId = points->InsertNextPoint(rayEnd(0), rayEnd(1), rayEnd(2));

        // Create a line connecting the start and end points
        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, startId);
        line->GetPointIds()->SetId(1, endId);

        // Add the line to the vtkCellArray
        lines->InsertNextCell(line);
    }

    // Create a vtkPolyData to hold the points and lines
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetLines(lines);

    // Create a mapper for the polydata
    vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    lineMapper->SetInputData(polyData);

    // Create an actor for the lines
    vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
    lineActor->SetMapper(lineMapper);
    lineActor->GetProperty()->SetColor(color(0), color(1), color(2));  // Set the color of the rays
    lineActor->GetProperty()->SetLineWidth(1.0);  // Set line thickness
    lineActor->GetProperty()->SetOpacity(1.0);  // Set the opacity for transparency


    // Add the actor to the renderer
    renderer->AddActor(lineActor);
}


void Visualizer::showRaysParallel(visioncraft::Viewpoint& viewpoint) {
    // Get the rays generated from the viewpoint
    auto rays = viewpoint.generateRays();

    // Get the viewpoint position
    Eigen::Vector3d viewpointPosition = viewpoint.getPosition();

    // Define a color palette with 12 distinct, visually appealing colors
   std::vector<Eigen::Vector3d> colors = {
        Eigen::Vector3d(0.0, 0.717, 0.925),  // Cyan (#00B7EB)
        Eigen::Vector3d(1.0, 0.0, 1.0),  // Magenta (#FF00FF)
        Eigen::Vector3d(0.224, 1.0, 0.078),  // Neon Green (#39FF14)
        Eigen::Vector3d(1.0, 1.0, 0.0),  // Bright Yellow (#FFFF00)
        Eigen::Vector3d(1.0, 0.373, 0.0),  // Neon Orange (#FF5F00)
        Eigen::Vector3d(1.0, 0.5, 0.0),  // Orange (#FF8000)
        Eigen::Vector3d(0.5, 0.0, 0.0),  // Red (#800000)
        Eigen::Vector3d(0.0, 0.5, 1.0),  // Bright Blue (#0080FF)
        Eigen::Vector3d(0.5, 1.0, 1.0),  // Light Cyan (#80FFFF)
        Eigen::Vector3d(0.8, 0.8, 0.0),  // Yellow Green (#CCCC00)
        Eigen::Vector3d(0.5, 0.0, 0.5),  // Purple (#800080)
        Eigen::Vector3d(1.0, 0.8, 0.8)   // Light Pink (#FFCCCC)
    };



    // Get the number of rays
    int numRays = rays.size();

    // Max number of threads (colors)
    int maxThreads = 12;

    // Calculate the batch size (round down if necessary)
    int batchSize = numRays / maxThreads;

    // Iterate over the colors and assign rays to each color batch
    for (int batch = 0; batch < maxThreads; ++batch) {
        // Create a new vtkPoints and vtkCellArray for each batch
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();

        // Calculate the range of rays for this batch
        int startIdx = batch * batchSize;
        int endIdx = (batch == maxThreads - 1) ? numRays : (startIdx + batchSize);  // Handle remaining rays

        // Get the current color for this batch
        Eigen::Vector3d currentColor = colors[batch];

        // Iterate over the rays for this batch
        for (int i = startIdx; i < endIdx; ++i) {
            const auto& rayEnd = rays[i];

            // Insert the start and end points of the ray
            vtkIdType startId = points->InsertNextPoint(viewpointPosition(0), viewpointPosition(1), viewpointPosition(2));
            vtkIdType endId = points->InsertNextPoint(rayEnd(0), rayEnd(1), rayEnd(2));

            // Create a line connecting the start and end points
            vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
            line->GetPointIds()->SetId(0, startId);
            line->GetPointIds()->SetId(1, endId);

            // Add the line to the vtkCellArray
            lines->InsertNextCell(line);
        }

        // Create a vtkPolyData to hold the points and lines
        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->SetPoints(points);
        polyData->SetLines(lines);

        // Create a mapper for the polydata
        vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        lineMapper->SetInputData(polyData);

        // Create a new actor for the lines and set the color
        vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
        lineActor->SetMapper(lineMapper);
        lineActor->GetProperty()->SetColor(currentColor(0), currentColor(1), currentColor(2));
        lineActor->GetProperty()->SetLineWidth(1.0);  // Set line thickness
        lineActor->GetProperty()->SetOpacity(1.0);  // Set the opacity for transparency

        // Add the actor to the renderer
        renderer->AddActor(lineActor);
    }
}


void Visualizer::showRayVoxels(visioncraft::Viewpoint& viewpoint, const std::shared_ptr<octomap::ColorOcTree>& octomap, const Eigen::Vector3d& color) {
    std::cout << "[INFO] Starting showRayVoxels for all rays..." << std::endl;

    // Get the rays generated from the viewpoint
    auto rays = viewpoint.generateRays();
    if (rays.empty()) {
        std::cout << "[ERROR] No rays generated!" << std::endl;
        return;
    }

    // Retrieve the voxel size from the octomap
    double voxel_size = octomap->getResolution();
    std::cout << "[INFO] Voxel size: " << voxel_size << std::endl;

    // Create an append filter to combine all the voxel cubes into a single dataset
    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

    // Color array for the voxels
    vtkSmartPointer<vtkUnsignedCharArray> voxelColors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    voxelColors->SetNumberOfComponents(3);  // RGB
    voxelColors->SetName("Colors");

    // Precompute the color once, as it doesn't change for each voxel
    unsigned char voxelColor[3] = {
        static_cast<unsigned char>(color(0) * 255),
        static_cast<unsigned char>(color(1) * 255),
        static_cast<unsigned char>(color(2) * 255)
    };

    // Loop through all rays
    int total_voxel_count = 0;
    for (const auto& rayEnd : rays) {
        // Get the viewpoint position
        Eigen::Vector3d viewpointPosition = viewpoint.getPosition();

        octomap::KeyRay keyRay;
        octomap::point3d ray_origin(viewpointPosition(0), viewpointPosition(1), viewpointPosition(2));
        octomap::point3d ray_end(rayEnd(0), rayEnd(1), rayEnd(2));

        // Ensure that raycasting proceeds through all voxels along the ray
        bool ray_cast_success = octomap->computeRayKeys(ray_origin, ray_end, keyRay);
        if (!ray_cast_success) {
            std::cout << "[ERROR] Failed to compute ray keys for one of the rays." << std::endl;
            continue; // Skip to the next ray
        }

        int voxel_count = 0;
        for (const auto& key : keyRay) {
            octomap::point3d voxel_center = octomap->keyToCoord(key);

            // Create a cube for each voxel
            vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
            cubeSource->SetCenter(voxel_center.x(), voxel_center.y(), voxel_center.z());
            cubeSource->SetXLength(voxel_size);
            cubeSource->SetYLength(voxel_size);
            cubeSource->SetZLength(voxel_size);
            cubeSource->Update(); // Update each cube

            // Add the cube to the append filter
            appendFilter->AddInputData(cubeSource->GetOutput());

            voxel_count++;
            total_voxel_count++;

            if (total_voxel_count % 1000 == 0) {
                std::cout << "[INFO] Processed " << total_voxel_count << " voxels so far..." << std::endl;
            }
        }
    }

    std::cout << "[INFO] Finished processing " << total_voxel_count << " voxels for all rays." << std::endl;

    // Finalize the appended polydata
    appendFilter->Update();

    // Get the combined polydata from the append filter
    vtkSmartPointer<vtkPolyData> voxelPolyData = appendFilter->GetOutput();
    voxelPolyData->GetPointData()->SetScalars(voxelColors); // Assign the colors

    // Create a mapper for the combined voxel polydata
    vtkSmartPointer<vtkPolyDataMapper> voxelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    voxelMapper->SetInputData(voxelPolyData);

    // Create an actor for the combined voxel cubes
    vtkSmartPointer<vtkActor> voxelActor = vtkSmartPointer<vtkActor>::New();
    voxelActor->SetMapper(voxelMapper);

    // Set the transparency and color for the cubes
    voxelActor->GetProperty()->SetOpacity(0.5);  // Set to 50% transparent
    voxelActor->GetProperty()->SetColor(color(0), color(1), color(2));  // Set color

    // Add the voxel actor to the renderer
    renderer->AddActor(voxelActor);

    std::cout << "[INFO] showRayVoxels completed and visualization updated." << std::endl;
}




void Visualizer::showViewpointHits(const std::shared_ptr<octomap::ColorOcTree>& octomap) {
    if (!octomap) return;

    // Create a vtkAppendPolyData to hold the polydata for all the cubes (voxels)
    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    
    // Array to store the colors for each voxel
    vtkSmartPointer<vtkUnsignedCharArray> voxelColors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    voxelColors->SetNumberOfComponents(3);  // RGB components
    voxelColors->SetName("Colors");  // Name the array

    // Iterate over all leaf nodes (voxels) in the octomap
    for (octomap::ColorOcTree::leaf_iterator it = octomap->begin_leafs(), end = octomap->end_leafs(); it != end; ++it) {
        if (octomap->isNodeOccupied(*it)) {
            // For each occupied voxel, create a VTK cube source representing the voxel
            vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
            cubeSource->SetCenter(it.getX(), it.getY(), it.getZ());  // Set the voxel's position
            cubeSource->SetXLength(octomap->getResolution());
            cubeSource->SetYLength(octomap->getResolution());
            cubeSource->SetZLength(octomap->getResolution());

            // Update the cube source
            cubeSource->Update();

            // Get the voxel's color from the octomap
            octomap::ColorOcTreeNode::Color voxelColor = it->getColor();

            // Convert the color to RGB format (0-255)
            unsigned char rgb[3] = {
                static_cast<unsigned char>(voxelColor.r),
                static_cast<unsigned char>(voxelColor.g),
                static_cast<unsigned char>(voxelColor.b)
            };

            // Get the number of points in the cube's polydata
            vtkSmartPointer<vtkPolyData> cubePolyData = cubeSource->GetOutput();
            vtkSmartPointer<vtkPoints> cubePoints = cubePolyData->GetPoints();

            // For each point in the cube (there are 8 points for a cube), set its color
            for (vtkIdType i = 0; i < cubePoints->GetNumberOfPoints(); ++i) {
                voxelColors->InsertNextTypedTuple(rgb);  // Assign the color to each point
            }

            // Append the cube's polydata to the appendFilter
            appendFilter->AddInputData(cubeSource->GetOutput());
        }
    }

    // Finalize the appended polydata (all cubes combined)
    appendFilter->Update();

    // Get the combined polydata (all voxels)
    vtkSmartPointer<vtkPolyData> voxelPolyData = appendFilter->GetOutput();

    // Set the colors for the polydata
    voxelPolyData->GetPointData()->SetScalars(voxelColors);

    // Create a mapper for the polydata
    vtkSmartPointer<vtkPolyDataMapper> voxelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    voxelMapper->SetInputData(voxelPolyData);

    // Create an actor for the octomap
    vtkSmartPointer<vtkActor> voxelActor = vtkSmartPointer<vtkActor>::New();
    voxelActor->SetMapper(voxelMapper);

    // Set the edge visibility and color (optional, to highlight edges)
    voxelActor->GetProperty()->SetEdgeVisibility(1);  // Enable edge visibility
    voxelActor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);  // Set edge color to black (neutral)
    voxelActor->GetProperty()->SetLineWidth(1.0);  // Set the width of the edges

    // Add the actor to the renderer
    renderer->AddActor(voxelActor);
}


void Visualizer::addMesh(const visioncraft::Model& model, const Eigen::Vector3d& color ) {
    auto mesh = model.getMeshData();
    if (!mesh) return;

    vtkSmartPointer<vtkPoints> vtkPointsData = vtkSmartPointer<vtkPoints>::New();
    for (const auto& vertex : mesh->vertices_) {
        vtkPointsData->InsertNextPoint(vertex.x(), vertex.y(), vertex.z());
    }

    vtkSmartPointer<vtkCellArray> vtkTriangles = vtkSmartPointer<vtkCellArray>::New();
    for (const auto& triangle : mesh->triangles_) {
        vtkIdType ids[3] = {triangle(0), triangle(1), triangle(2)};
        vtkTriangles->InsertNextCell(3, ids);
    }

    vtkSmartPointer<vtkPolyData> vtkMesh = vtkSmartPointer<vtkPolyData>::New();
    vtkMesh->SetPoints(vtkPointsData);
    vtkMesh->SetPolys(vtkTriangles);

    vtkSmartPointer<vtkPolyDataMapper> meshMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    meshMapper->SetInputData(vtkMesh);

    vtkSmartPointer<vtkActor> meshActor = vtkSmartPointer<vtkActor>::New();
    meshActor->SetMapper(meshMapper);
    meshActor->GetProperty()->SetColor(color(0), color(1), color(2));  // Set mesh color
    renderer->AddActor(meshActor);
}



void Visualizer::addPointCloud(const visioncraft::Model& model, const Eigen::Vector3d& color ) {
    auto pointCloud = model.getPointCloud();
    if (!pointCloud) return;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (const auto& point : pointCloud->points_) {
        points->InsertNextPoint(point(0), point(1), point(2));
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);

    vtkSmartPointer<vtkVertexGlyphFilter> glyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    glyphFilter->SetInputData(polyData);
    glyphFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(glyphFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color(0), color(1), color(2));  // Set point cloud color
    renderer->AddActor(actor);
}

void Visualizer::addOctomap(const visioncraft::Model& model, const Eigen::Vector3d& defaultColor) {
    auto octomap = model.getSurfaceShellOctomap();
    if (!octomap) return;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);  // RGB
    colors->SetName("Colors");

    int voxelCount = 0;
    for (octomap::ColorOcTree::leaf_iterator it = octomap->begin_leafs(), end = octomap->end_leafs(); it != end; ++it) {
        if (octomap->isNodeOccupied(*it)) {
            points->InsertNextPoint(it.getX(), it.getY(), it.getZ());
            voxelCount++;

            // Assign voxel-specific color or default color
            Eigen::Vector3d color = (defaultColor(0) >= 0 && defaultColor(1) >= 0 && defaultColor(2) >= 0)
                                    ? defaultColor
                                    : Eigen::Vector3d(it->getColor().r / 255.0, it->getColor().g / 255.0, it->getColor().b / 255.0);

            unsigned char voxelColor[3] = {
                static_cast<unsigned char>(color(0) * 255),
                static_cast<unsigned char>(color(1) * 255),
                static_cast<unsigned char>(color(2) * 255)
            };
            colors->InsertNextTypedTuple(voxelColor);
        }
    }
    // std::cout << "[INFO] Number of occupied voxels: " << voxelCount << std::endl;

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->GetPointData()->SetScalars(colors);

    // Create a cube glyph with proper scaling
    vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
    cubeSource->SetXLength(octomap->getResolution());
    cubeSource->SetYLength(octomap->getResolution());
    cubeSource->SetZLength(octomap->getResolution());

    // Configure the glyph filter
    vtkSmartPointer<vtkGlyph3D> glyphFilter = vtkSmartPointer<vtkGlyph3D>::New();
    glyphFilter->SetInputData(polyData);
    glyphFilter->SetSourceConnection(cubeSource->GetOutputPort());
    glyphFilter->SetColorModeToColorByScalar();  // Apply colors per voxel
    glyphFilter->SetScaleModeToDataScalingOff();  // Avoid additional scaling
    glyphFilter->Update();

    // Set up mapper and actor for rendering
    vtkSmartPointer<vtkPolyDataMapper> voxelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    voxelMapper->SetInputConnection(glyphFilter->GetOutputPort());
    voxelMapper->SetScalarModeToUsePointData();  // Use colors from point data

    vtkSmartPointer<vtkActor> octomapActor = vtkSmartPointer<vtkActor>::New();
    octomapActor->SetMapper(voxelMapper);
    octomapActor->GetProperty()->SetEdgeVisibility(1);
    octomapActor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
    octomapActor->GetProperty()->SetLineWidth(1.0);

    // Reset camera and add actor to renderer
    renderer->AddActor(octomapActor);


    // std::cout << "[INFO] Octomap visualization complete." << std::endl;
}



void Visualizer::showGPUVoxelGrid(const visioncraft::Model& model, const Eigen::Vector3d& color) {
    const auto& gpuVoxelGrid = model.getGPUVoxelGrid();

    if (!gpuVoxelGrid.voxel_data) {
        std::cerr << "[ERROR] Voxel data in GPU format is not available." << std::endl;
        return;
    }

    // Retrieve voxel grid dimensions and size
    int width = gpuVoxelGrid.width;
    int height = gpuVoxelGrid.height;
    int depth = gpuVoxelGrid.depth;
    double voxelSize = gpuVoxelGrid.voxel_size;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    // Collect all occupied voxel centers
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int linear_idx = z * (width * height) + y * width + x;

                if (gpuVoxelGrid.voxel_data[linear_idx] == 1) {
                    double x_center = gpuVoxelGrid.min_bound[0] + x * voxelSize;
                    double y_center = gpuVoxelGrid.min_bound[1] + y * voxelSize;
                    double z_center = gpuVoxelGrid.min_bound[2] + z * voxelSize;

                    points->InsertNextPoint(x_center, y_center, z_center);
                }
            }
        }
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);

    // Create a cube glyph for each occupied voxel
    vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
    cubeSource->SetXLength(voxelSize);
    cubeSource->SetYLength(voxelSize);
    cubeSource->SetZLength(voxelSize);

    vtkSmartPointer<vtkGlyph3D> glyphFilter = vtkSmartPointer<vtkGlyph3D>::New();
    glyphFilter->SetInputData(polyData);
    glyphFilter->SetSourceConnection(cubeSource->GetOutputPort());
    glyphFilter->SetColorModeToColorByScalar();  // Coloring will be applied uniformly in actor properties
    glyphFilter->SetScaleModeToDataScalingOff();  // Prevents any additional scaling of the glyphs
    glyphFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> voxelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    voxelMapper->SetInputConnection(glyphFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> voxelActor = vtkSmartPointer<vtkActor>::New();
    voxelActor->SetMapper(voxelMapper);
    voxelActor->GetProperty()->SetColor(color(0), color(1), color(2));  // Apply uniform color from parameter
    voxelActor->GetProperty()->SetOpacity(0.5);  // Optional transparency for visual effect

    // Add the voxel actor to the renderer
    renderer->AddActor(voxelActor);

    // std::cout << "[INFO] VoxelGridGPU visualization added to renderer." << std::endl;
}


void Visualizer::addVoxelMap(const visioncraft::Model& model, const Eigen::Vector3d& defaultColor) {
    const auto& metaVoxelMap = model.getVoxelMap().getMap(); // Access the internal map with getMap
    if (metaVoxelMap.empty()) return;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);  // RGB
    colors->SetName("Colors");

    int voxelCount = 0;
    for (const auto& kv : metaVoxelMap) {
        const auto& metaVoxel = kv.second;
        const auto& voxelPos = metaVoxel.getPosition();  // Use MetaVoxel's stored position
        points->InsertNextPoint(voxelPos.x(), voxelPos.y(), voxelPos.z());
        voxelCount++;

        // Use defaultColor if specified, otherwise color based on occupancy property if available
        Eigen::Vector3d color = (defaultColor(0) >= 0 && defaultColor(1) >= 0 && defaultColor(2) >= 0)
                                ? defaultColor
                                : Eigen::Vector3d(metaVoxel.getOccupancy(), 0.0, 1.0 - metaVoxel.getOccupancy());

        unsigned char voxelColor[3] = {
            static_cast<unsigned char>(color(0) * 255),
            static_cast<unsigned char>(color(1) * 255),
            static_cast<unsigned char>(color(2) * 255)
        };
        colors->InsertNextTypedTuple(voxelColor);
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->GetPointData()->SetScalars(colors);

    vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
    cubeSource->SetXLength(model.getOctomap()->getResolution());
    cubeSource->SetYLength(model.getOctomap()->getResolution());
    cubeSource->SetZLength(model.getOctomap()->getResolution());

    vtkSmartPointer<vtkGlyph3D> glyphFilter = vtkSmartPointer<vtkGlyph3D>::New();
    glyphFilter->SetInputData(polyData);
    glyphFilter->SetSourceConnection(cubeSource->GetOutputPort());
    glyphFilter->SetColorModeToColorByScalar();
    glyphFilter->SetScaleModeToDataScalingOff();
    glyphFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> voxelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    voxelMapper->SetInputConnection(glyphFilter->GetOutputPort());
    voxelMapper->SetScalarModeToUsePointData();

    vtkSmartPointer<vtkActor> voxelActor = vtkSmartPointer<vtkActor>::New();
    voxelActor->SetMapper(voxelMapper);
    voxelActor->GetProperty()->SetEdgeVisibility(1);
    voxelActor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
    voxelActor->GetProperty()->SetLineWidth(1.0);

    renderer->AddActor(voxelActor);
}

void Visualizer::addVoxelMapProperty(const visioncraft::Model& model, const std::string& property_name, 
                                     const Eigen::Vector3d& baseColor, const Eigen::Vector3d& propertyColor, 
                                     float minScale, float maxScale) {
    const auto& metaVoxelMap = model.getVoxelMap().getMap();
    if (metaVoxelMap.empty()) return;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);  // RGB
    colors->SetName("Colors");

    // If minScale and maxScale are not provided, calculate them from the property values
    if (minScale == -1.0f || maxScale == -1.0f) {
        minScale = std::numeric_limits<float>::max();
        maxScale = std::numeric_limits<float>::lowest();
        for (const auto& kv : metaVoxelMap) {
            const auto& metaVoxel = kv.second;
            if (metaVoxel.hasProperty(property_name)) {
                try {
                    float value = 0.0f;
                    const auto& prop = metaVoxel.getProperty(property_name);
                    if (prop.type() == typeid(int)) {
                        value = static_cast<float>(boost::get<int>(prop));
                    } else if (prop.type() == typeid(float)) {
                        value = boost::get<float>(prop);
                    } else if (prop.type() == typeid(double)) {
                        value = static_cast<float>(boost::get<double>(prop)); 
                    }
                    minScale = std::min(minScale, value);
                    maxScale = std::max(maxScale, value);
                } catch (const boost::bad_get& e) {
                    std::cerr << "Warning: Property " << property_name 
                              << " has an unexpected type for voxel at position " 
                              << metaVoxel.getPosition().transpose() << ": " << e.what() << std::endl;
                    continue;
                }
            }
        }
    }

    auto rainbowColorMap = [](float normalizedValue) -> Eigen::Vector3d {
        // Compute RGB using a rainbow color scheme (Blue -> Green -> Red)
        float r = std::max(0.0f, std::min(1.0f, -4.0f * std::abs(normalizedValue - 0.75f) + 1.5f));
        float g = std::max(0.0f, std::min(1.0f, -4.0f * std::abs(normalizedValue - 0.5f) + 1.5f));
        float b = std::max(0.0f, std::min(1.0f, -4.0f * std::abs(normalizedValue - 0.25f) + 1.5f));
        return Eigen::Vector3d(r, g, b);
    };

    for (const auto& kv : metaVoxelMap) {
        const auto& metaVoxel = kv.second;
        const auto& voxelPos = metaVoxel.getPosition();
        points->InsertNextPoint(voxelPos.x(), voxelPos.y(), voxelPos.z());

        Eigen::Vector3d color(0.0, 0.0, 1.0);  // Default to blue for undefined values

        if (metaVoxel.hasProperty(property_name)) {
            try {
                float propertyValue = 0.0f;
                const auto& prop = metaVoxel.getProperty(property_name);

                if (prop.type() == typeid(int)) {
                    propertyValue = static_cast<float>(boost::get<int>(prop));
                } else if (prop.type() == typeid(float)) {
                    propertyValue = boost::get<float>(prop);
                } else if (prop.type() == typeid(double)) {
                    propertyValue = static_cast<float>(boost::get<double>(prop));
                }

                float normalizedValue = (propertyValue - minScale) / (maxScale - minScale);
                normalizedValue = std::max(0.0f, std::min(normalizedValue, 1.0f));
                color = rainbowColorMap(normalizedValue);

            } catch (const boost::bad_get& e) {
                std::cerr << "Error: Failed to retrieve property " << property_name 
                          << " for voxel at position " << voxelPos.transpose() 
                          << ": " << e.what() << std::endl;
                continue;
            }
        }

        unsigned char voxelColor[3] = {
            static_cast<unsigned char>(color(0) * 255),
            static_cast<unsigned char>(color(1) * 255),
            static_cast<unsigned char>(color(2) * 255)
        };
        colors->InsertNextTypedTuple(voxelColor);
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->GetPointData()->SetScalars(colors);

    vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
    cubeSource->SetXLength(model.getOctomap()->getResolution());
    cubeSource->SetYLength(model.getOctomap()->getResolution());
    cubeSource->SetZLength(model.getOctomap()->getResolution());

    vtkSmartPointer<vtkGlyph3D> glyphFilter = vtkSmartPointer<vtkGlyph3D>::New();
    glyphFilter->SetInputData(polyData);
    glyphFilter->SetSourceConnection(cubeSource->GetOutputPort());
    glyphFilter->SetColorModeToColorByScalar();
    glyphFilter->SetScaleModeToDataScalingOff();
    glyphFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> voxelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    voxelMapper->SetInputConnection(glyphFilter->GetOutputPort());
    voxelMapper->SetScalarModeToUsePointData();

    vtkSmartPointer<vtkActor> voxelActor = vtkSmartPointer<vtkActor>::New();
    voxelActor->SetMapper(voxelMapper);
    voxelActor->GetProperty()->SetEdgeVisibility(1);
    voxelActor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
    voxelActor->GetProperty()->SetLineWidth(1.0);

    voxelMapPropertyActor_ = voxelActor;  // Store the actor for later removal
    renderer->AddActor(voxelMapPropertyActor_);
}



void Visualizer::removeVoxelMapProperty() {
    if (voxelMapPropertyActor_) {
        renderer->RemoveActor(voxelMapPropertyActor_);  // Remove from renderer
        voxelMapPropertyActor_ = nullptr;  // Reset the actor pointer
    }
}


void Visualizer::setBackgroundColor(const Eigen::Vector3d& color) {
    renderer->SetBackground(color(0), color(1), color(2));
}

void Visualizer::setViewpointFrustumColor(const Eigen::Vector3d& color) {
    // Implementation to update frustum color (if applicable).
}

void Visualizer::render() {
    // Initialize rendering components if necessary
    if (!renderWindowInteractor->GetInitialized()) {
        renderWindowInteractor->Initialize();
    }

    // Render the scene
    renderWindow->Render();

    // Process interaction events (non-blocking)
    renderWindowInteractor->ProcessEvents();
}



void visioncraft::Visualizer::renderStep() {
    renderWindow->Render();                // Update the scene
    renderWindowInteractor->ProcessEvents();  // Handle interactions
}



std::vector<vtkSmartPointer<vtkActor>> Visualizer::showFrustum(const visioncraft::Viewpoint& viewpoint) {
    std::vector<vtkSmartPointer<vtkActor>> frustumActors;

    // Get the frustum corners from the viewpoint
    std::vector<Eigen::Vector3d> corners = viewpoint.getFrustumCorners();
    if (corners.size() != 8) return frustumActors; // Return empty if corners are invalid

    auto createFrustumLine = [&](const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& color) {
        vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
        lineSource->SetPoint1(start(0), start(1), start(2));
        lineSource->SetPoint2(end(0), end(1), end(2));

        vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        lineMapper->SetInputConnection(lineSource->GetOutputPort());

        vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
        lineActor->SetMapper(lineMapper);
        lineActor->GetProperty()->SetColor(color(0), color(1), color(2));
        lineActor->GetProperty()->SetLineWidth(0.5);

        renderer->AddActor(lineActor);
        return lineActor;
    };

    Eigen::Vector3d frustumColor(1.0, 1.0, 0.0);  // Yellow for the frustum

    for (int i = 0; i < 4; ++i) {
        frustumActors.push_back(createFrustumLine(corners[i], corners[(i + 1) % 4], frustumColor));
        frustumActors.push_back(createFrustumLine(corners[i + 4], corners[(i + 1) % 4 + 4], frustumColor));
        frustumActors.push_back(createFrustumLine(corners[i], corners[i + 4], frustumColor));
    }

    return frustumActors;
}

std::vector<vtkSmartPointer<vtkActor>> Visualizer::showAxes(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation) {
    std::vector<vtkSmartPointer<vtkActor>> axesActors;

    auto createAxis = [](const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& color) {
        vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
        lineSource->SetPoint1(start(0), start(1), start(2));
        lineSource->SetPoint2(end(0), end(1), end(2));

        vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        lineMapper->SetInputConnection(lineSource->GetOutputPort());

        vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
        lineActor->SetMapper(lineMapper);
        lineActor->GetProperty()->SetColor(color(0), color(1), color(2));
        lineActor->GetProperty()->SetLineWidth(2.0);

        return lineActor;
    };

    double axisLength = 5.0;

    vtkSmartPointer<vtkActor> xAxis = createAxis(position, position + axisLength * orientation.col(0), Eigen::Vector3d(1.0, 0.0, 0.0));
    vtkSmartPointer<vtkActor> yAxis = createAxis(position, position + axisLength * orientation.col(1), Eigen::Vector3d(0.0, 1.0, 0.0));
    vtkSmartPointer<vtkActor> zAxis = createAxis(position, position + axisLength * orientation.col(2), Eigen::Vector3d(0.0, 0.0, 1.0));

    renderer->AddActor(xAxis);
    renderer->AddActor(yAxis);
    renderer->AddActor(zAxis);

    axesActors.push_back(xAxis);
    axesActors.push_back(yAxis);
    axesActors.push_back(zAxis);

    return axesActors;
}


void Visualizer::addOverlayText(const std::string& text, double x, double y, int fontSize, const Eigen::Vector3d& color) {
    // Create a text actor for the overlay text
    vtkSmartPointer<vtkTextActor> textActor = vtkSmartPointer<vtkTextActor>::New();
    textActor->SetInput(text.c_str());

    // Position the text using normalized viewport coordinates
    textActor->SetPosition(x * renderWindow->GetSize()[0], y * renderWindow->GetSize()[1]);

    // Configure text properties
    vtkTextProperty* textProperty = textActor->GetTextProperty();
    textProperty->SetFontSize(fontSize);
    textProperty->SetColor(color(0), color(1), color(2)); // Set the text color
    textProperty->SetJustificationToLeft();
    textProperty->SetVerticalJustificationToBottom();

    // Add the text actor to the renderer
    renderer->AddActor2D(textActor);

    // Store the text actor for future removal
    overlayTextActors_.push_back(textActor);
}

void Visualizer::removeOverlayTexts() {
    // Remove each overlay text actor from the renderer
    for (auto& actor : overlayTextActors_) {
        renderer->RemoveActor2D(actor);
    }
    overlayTextActors_.clear(); // Clear the vector to remove all references
}


} // namespace visioncraft
