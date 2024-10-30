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

namespace visioncraft {


Visualizer::Visualizer() {
    renderer = vtkSmartPointer<vtkRenderer>::New();
    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

    renderWindow->SetSize(1280, 720);
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

void Visualizer::addViewpoint(const visioncraft::Viewpoint& viewpoint, bool showFrustum, bool showAxes) {
    Eigen::Vector3d position = viewpoint.getPosition();
    Eigen::Matrix3d orientationMatrix = viewpoint.getOrientationMatrix();

    // Only show the axes for now, frustum will be added later
    if (showAxes) {
        this->showAxes(position, orientationMatrix);  // Now passing the orientation matrix instead of quaternion
    }


    // Show frustum
    if (showFrustum) {
        this->showFrustum(viewpoint);  // Visualize frustum
    }
}

void Visualizer::addMultipleViewpoints(const std::vector<visioncraft::Viewpoint>& viewpoints) {
    for (const auto& viewpoint : viewpoints) {
        addViewpoint(viewpoint);
    }
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
                    float value = (metaVoxel.getProperty(property_name).type() == typeid(int))
                        ? static_cast<float>(boost::get<int>(metaVoxel.getProperty(property_name)))
                        : boost::get<float>(metaVoxel.getProperty(property_name));
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

    for (const auto& kv : metaVoxelMap) {
        const auto& metaVoxel = kv.second;
        const auto& voxelPos = metaVoxel.getPosition();  // Use MetaVoxel's stored position
        points->InsertNextPoint(voxelPos.x(), voxelPos.y(), voxelPos.z());

        // Default to base color if property is missing
        Eigen::Vector3d color = baseColor;

        // Adjust color intensity based on the property value
        if (metaVoxel.hasProperty(property_name)) {
            try {
                float propertyValue = (metaVoxel.getProperty(property_name).type() == typeid(int))
                    ? static_cast<float>(boost::get<int>(metaVoxel.getProperty(property_name)))
                    : boost::get<float>(metaVoxel.getProperty(property_name));

                // Normalize property value to [0,1]
                float normalizedValue = (propertyValue - minScale) / (maxScale - minScale);
                normalizedValue = std::max(0.0f, std::min(normalizedValue, 1.0f));


                // Interpolate between baseColor and propertyColor
                color = baseColor * (1.0f - normalizedValue) + propertyColor * normalizedValue;
                color = color.cwiseMin(1.0).cwiseMax(0.0); // Ensure color stays within [0,1] range
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

    renderer->AddActor(voxelActor);
}



void Visualizer::setBackgroundColor(const Eigen::Vector3d& color) {
    renderer->SetBackground(color(0), color(1), color(2));
}

void Visualizer::setViewpointFrustumColor(const Eigen::Vector3d& color) {
    // Implementation to update frustum color (if applicable).
}

void Visualizer::render() {
        // Initialize the window interactor before starting the rendering loop
    renderWindowInteractor->Initialize();
    
    // Render the window and start the interaction
    renderWindow->Render();
    
    // Start the interaction loop
    renderWindowInteractor->Start();
}



void Visualizer::showFrustum(const visioncraft::Viewpoint& viewpoint) {
    // Get the frustum corners from the viewpoint
    std::vector<Eigen::Vector3d> corners = viewpoint.getFrustumCorners();

    // Ensure there are exactly 8 corners (4 for near plane, 4 for far plane)
    if (corners.size() != 8) return;

    // Function to create lines between two points
    auto createFrustumLine = [&](const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& color) {
        vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
        lineSource->SetPoint1(start(0), start(1), start(2));
        lineSource->SetPoint2(end(0), end(1), end(2));

        vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        lineMapper->SetInputConnection(lineSource->GetOutputPort());

        vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
        lineActor->SetMapper(lineMapper);
        lineActor->GetProperty()->SetColor(color(0), color(1), color(2));
        lineActor->GetProperty()->SetLineWidth(0.5);  // Set line thickness

        renderer->AddActor(lineActor);
    };

    // Define color for the frustum lines
    Eigen::Vector3d frustumColor(1.0, 1.0, 0.0);  // Yellow for the frustum

    // Draw lines between near and far plane corners
    for (int i = 0; i < 4; ++i) {
        // Near plane edges
        createFrustumLine(corners[i], corners[(i + 1) % 4], frustumColor);
        // Far plane edges
        createFrustumLine(corners[i + 4], corners[(i + 1) % 4 + 4], frustumColor);
        // Lines connecting near and far planes
        createFrustumLine(corners[i], corners[i + 4], frustumColor);
    }
}

void Visualizer::showAxes(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation) {
    // Function to create a line for each axis
    auto createAxis = [](const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& color) {
        vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
        lineSource->SetPoint1(start(0), start(1), start(2));
        lineSource->SetPoint2(end(0), end(1), end(2));

        vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        lineMapper->SetInputConnection(lineSource->GetOutputPort());

        vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
        lineActor->SetMapper(lineMapper);
        lineActor->GetProperty()->SetColor(color(0), color(1), color(2));
        lineActor->GetProperty()->SetLineWidth(2.0);  // Set line thickness

        return lineActor;
    };

    // Define axis length
    double axisLength = 5.0;

    // Create X, Y, Z axes
    vtkSmartPointer<vtkActor> xAxis = createAxis(position, position + axisLength * orientation.col(0), Eigen::Vector3d(1.0, 0.0, 0.0)); // Red for X
    vtkSmartPointer<vtkActor> yAxis = createAxis(position, position + axisLength * orientation.col(1), Eigen::Vector3d(0.0, 1.0, 0.0)); // Green for Y
    vtkSmartPointer<vtkActor> zAxis = createAxis(position, position + axisLength * orientation.col(2), Eigen::Vector3d(0.0, 0.0, 1.0)); // Blue for Z

    // Add actors to the renderer
    renderer->AddActor(xAxis);
    renderer->AddActor(yAxis);
    renderer->AddActor(zAxis);
}



} // namespace visioncraft
