#include <vtkSmartPointer.h>        // For smart pointers to VTK objects
#include <vtkCubeSource.h>          // For generating cubes representing voxels
#include <vtkPolyDataMapper.h>      // For mapping polygonal data (such as cubes) to graphics primitives
#include <vtkActor.h>               // For representing an entity in the rendering scene
#include <vtkProperty.h>            // For setting visual properties of an actor
#include <vtkRenderer.h>            // For managing rendering process
#include <vtkRenderWindow.h>        // For rendering window
#include <vtkRenderWindowInteractor.h> // For interaction with the render window
#include <vtkAppendPolyData.h>      // For combining multiple polydata objects
#include <vtkBoundingBox.h>         // For calculating bounding box of 3D data
#include <fstream>                  // For file operations
#include <sstream>                  // For string stream operations
#include <string>                   // For handling strings
#include <iostream>                 // For input-output stream operations
#include <vector>                   // For using dynamic arrays
#include <algorithm>                // For common algorithms like std::max

#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkRenderingOpenGL);
VTK_MODULE_INIT(vtkInteractionStyle);


int main(int argc, char *argv[])
{
    // Path to the CSV file containing voxel data
    std::string filePath = "../data/exploration_map.csv";

    // Vectors to store the x, y, z coordinates and voxel values from the CSV file
    std::vector<double> xCoords, yCoords, zCoords, voxelValues;

    // Attempt to open the CSV file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        // Print an error message and exit if the file cannot be opened
        std::cerr << "Error opening file!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;

    // Skip the header line
    std::getline(file, line);

    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x, y, z, value;

        // Extract x, y, z, and value components separated by commas
        if (!(std::getline(ss, x, ',') && std::getline(ss, y, ',') &&
              std::getline(ss, z, ',') && std::getline(ss, value, ','))) {
            // Print an error message and skip the line if it cannot be parsed
            std::cerr << "Error parsing line: " << line << std::endl;
            continue; 
        }

        try {
            // Convert the string values to doubles
            double xPos = std::stod(x);
            double yPos = std::stod(y);
            double zPos = std::stod(z);
            double voxelValue = std::stod(value);

            // Store the coordinates and voxel value in their respective vectors
            xCoords.push_back(xPos);
            yCoords.push_back(yPos);
            zCoords.push_back(zPos);
            voxelValues.push_back(voxelValue);

        } catch (const std::invalid_argument& e) {
            // Handle cases where conversion to double fails due to invalid input
            std::cerr << "Invalid argument: " << e.what() << " in line: " << line << std::endl;
            continue; // Skip the malformed line
        } catch (const std::out_of_range& e) {
            // Handle cases where the input values are out of range for a double
            std::cerr << "Out of range error: " << e.what() << " in line: " << line << std::endl;
            continue; // Skip the malformed line
        }
    }
    // Close the file after reading all lines
    file.close();

    // Create a bounding box object to calculate the bounds of all voxels
    vtkBoundingBox boundingBox;
    for (size_t i = 0; i < xCoords.size(); ++i) {
        // Add each voxel's center point to the bounding box calculation
        boundingBox.AddPoint(xCoords[i], yCoords[i], zCoords[i]);
    }

    // Array to store the calculated bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    double bounds[6];
    boundingBox.GetBounds(bounds); // Retrieve the bounds of all points

    // Calculate the voxel size based on the bounds divided by a grid size (e.g., 32)
    double voxelSizeX = (bounds[1] - bounds[0]) / 32.0;
    double voxelSizeY = (bounds[3] - bounds[2]) / 32.0;
    double voxelSizeZ = (bounds[5] - bounds[4]) / 32.0;

    // Print the calculated voxel sizes for debugging purposes
    std::cout << "Voxel size: " << voxelSizeX << ", " << voxelSizeY << ", " << voxelSizeZ << std::endl;

    // Use the maximum voxel size to ensure all cubes fit within the bounds
    double voxelSize = std::max({voxelSizeX, voxelSizeY, voxelSizeZ}) * 33.0 / 32.0;

    // Append filters to combine multiple voxel polydata into single objects for rendering
    vtkSmartPointer<vtkAppendPolyData> appendFilterOccupied = vtkSmartPointer<vtkAppendPolyData>::New();
    vtkSmartPointer<vtkAppendPolyData> appendFilterUnknown = vtkSmartPointer<vtkAppendPolyData>::New();

    // Loop through all voxel coordinates and values
    for (size_t i = 0; i < xCoords.size(); ++i) {
        double xPos = xCoords[i];
        double yPos = yCoords[i];
        double zPos = zCoords[i];
        double voxelValue = voxelValues[i];

        // Skip empty voxels (voxelValue == 0.0) in visualization
        if (voxelValue == 0.0) {
            continue;
        }

        // Create a cube source to represent the voxel
        vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
        cubeSource->SetXLength(voxelSize);
        cubeSource->SetYLength(voxelSize);
        cubeSource->SetZLength(voxelSize);
        cubeSource->SetCenter(xPos, yPos, zPos);

        // Add the cube to the appropriate append filter based on voxel value
        if (voxelValue == 0.5) {
            appendFilterUnknown->AddInputConnection(cubeSource->GetOutputPort());
        } else if (voxelValue == 1.0) {
            appendFilterOccupied->AddInputConnection(cubeSource->GetOutputPort());
        }
    }

    // Mapper and Actor for occupied voxels (voxelValue == 1.0)
    vtkSmartPointer<vtkPolyDataMapper> mapperOccupied = vtkSmartPointer<vtkPolyDataMapper>::New();
    if (appendFilterOccupied->GetNumberOfInputConnections(0) > 0) {
        mapperOccupied->SetInputConnection(appendFilterOccupied->GetOutputPort());
    } else {
        std::cerr << "No occupied voxels to visualize." << std::endl;
    }

    vtkSmartPointer<vtkActor> actorOccupied = vtkSmartPointer<vtkActor>::New();
    actorOccupied->SetMapper(mapperOccupied);
    actorOccupied->GetProperty()->SetColor(0.0, 0.0, 1.0); // Blue color for occupied voxels
    actorOccupied->GetProperty()->SetEdgeVisibility(1);     // Enable edge visibility
    actorOccupied->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0); // Set edge color to black

    // Mapper and Actor for unknown voxels (voxelValue == 0.5)
    vtkSmartPointer<vtkPolyDataMapper> mapperUnknown = vtkSmartPointer<vtkPolyDataMapper>::New();
    if (appendFilterUnknown->GetNumberOfInputConnections(0) > 0) {
        mapperUnknown->SetInputConnection(appendFilterUnknown->GetOutputPort());
    } else {
        std::cerr << "No unknown voxels to visualize." << std::endl;
    }

    vtkSmartPointer<vtkActor> actorUnknown = vtkSmartPointer<vtkActor>::New();
    actorUnknown->SetMapper(mapperUnknown);
    actorUnknown->GetProperty()->SetColor(1.0, 1.0, 0.0); // Yellow color for unknown voxels
    actorUnknown->GetProperty()->SetEdgeVisibility(1);     // Enable edge visibility
    actorUnknown->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0); // Set edge color to black

    // Create a renderer and add the actors for occupied and unknown voxels
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    if (appendFilterOccupied->GetNumberOfInputConnections(0) > 0) {
        renderer->AddActor(actorOccupied);
    }
    if (appendFilterUnknown->GetNumberOfInputConnections(0) > 0) {
        renderer->AddActor(actorUnknown);
    }
    renderer->SetBackground(0.0, 0.0, 0.0); // Set background color to black

    // Create a render window and add the renderer to it
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(800, 800); // Set the window size

    // Create a render window interactor for handling user input and interaction
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Render the scene and start the interaction loop
    renderWindow->Render();
    renderWindowInteractor->Start();

    // Exit successfully
    return EXIT_SUCCESS;
}
