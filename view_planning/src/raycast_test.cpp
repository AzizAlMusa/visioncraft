#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>

class GridModel {
public:
    int gridSize;
    std::vector<std::vector<bool>> gridModel;

    GridModel(int size) : gridSize(size), gridModel(size, std::vector<bool>(size, false)) {}

    bool isOccupied(int x, int y) const {
        return gridModel[x][y];
    }

    void setOccupied(int x, int y, bool occupied) {
        if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
            gridModel[x][y] = occupied;
        }
    }

    void reset() {
        for (auto& row : gridModel) {
            std::fill(row.begin(), row.end(), false);
        }
    }
};



class Ray {
public:
    sf::Vector2f start;
    sf::Vector2f end;

    Ray(const sf::Vector2f& startPoint, const sf::Vector2f& endPoint) : start(startPoint), end(endPoint) {}

    std::vector<std::pair<int, int>> trace(const GridModel& model, int squareSize) const {
        std::vector<std::pair<int, int>> collision;

        float dx = end.x - start.x;
        float dy = end.y - start.y;
        float maxStep = std::max(std::abs(dx), std::abs(dy));
        float xIncrement = dx / maxStep;
        float yIncrement = dy / maxStep;

        float x = start.x;
        float y = start.y;

        for (int step = 0; step <= maxStep; step++) {
            int gridX = static_cast<int>(x) / squareSize;
            int gridY = static_cast<int>(y) / squareSize;

            if (gridX >= 0 && gridX < model.gridSize && gridY >= 0 && gridY < model.gridSize) {
                collision.emplace_back(gridX, gridY);

                if (model.isOccupied(gridX, gridY)) {
                    break;  // Stop tracing when a wall is hit
                }
            }

            x += xIncrement;
            y += yIncrement;
        }

        return collision;
    }
};



class GridVisualizer {
    sf::RenderWindow& window;
    GridModel& model;
    int squareSize;
    std::vector<std::vector<sf::RectangleShape>> gridShapes;

public:
    GridVisualizer(sf::RenderWindow& win, GridModel& mod, int sqSize) 
        : window(win), model(mod), squareSize(sqSize) {
        initializeGrid();
    }

    int getSquareSize() const {
        return squareSize;
    }

    void initializeGrid() {
        gridShapes.resize(model.gridSize, std::vector<sf::RectangleShape>(model.gridSize));
        for (int i = 0; i < model.gridSize; ++i) {
            for (int j = 0; j < model.gridSize; ++j) {
                gridShapes[i][j].setSize(sf::Vector2f(squareSize, squareSize));
                gridShapes[i][j].setOutlineColor(sf::Color::Black);
                gridShapes[i][j].setOutlineThickness(1.f);
                gridShapes[i][j].setPosition(i * squareSize, j * squareSize);
            }
        }
    }

    void draw() {
        window.clear(sf::Color::White);
        for (int i = 0; i < model.gridSize; ++i) {
            for (int j = 0; j < model.gridSize; ++j) {
                gridShapes[i][j].setFillColor(model.isOccupied(i, j) ? sf::Color::Blue : sf::Color::White);
                window.draw(gridShapes[i][j]);
            }
        }
        window.display();
    }
};


class EventHandler {
public:
    GridModel& gridModel;
    GridVisualizer& gridVisualizer;
    sf::RenderWindow& window;
    std::vector<Ray>& rays;
    sf::Vector2f& start;
    sf::Vector2f& end;
    bool& isTracing;

    EventHandler(sf::RenderWindow& win, GridModel& model, GridVisualizer& visualizer, std::vector<Ray>& r, sf::Vector2f& s, sf::Vector2f& e, bool& tracing)
        : window(win), gridModel(model), gridVisualizer(visualizer), rays(r), start(s), end(e), isTracing(tracing) {}

    void processEvents() {
        sf::Event event;
        while (window.pollEvent(event)) {
            switch (event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::MouseButtonPressed:
                    handleMouseButtonPressed(event);
                    break;
                case sf::Event::MouseButtonReleased:
                    handleMouseButtonReleased(event);
                    break;
                case sf::Event::KeyPressed:
                    handleKeyPressed(event);
                    break;
                // Add more cases as needed for other event types
            }
        }
    }

private:

    void handleMouseButtonPressed(const sf::Event& event) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            start = window.mapPixelToCoords(sf::Mouse::getPosition(window));
            isTracing = true;
        } else if (event.mouseButton.button == sf::Mouse::Right) {
            // Check if the right mouse button is being held down
            while (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
                sf::Vector2f gridPos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
                int gridX = static_cast<int>(gridPos.x) / gridVisualizer.getSquareSize();
                int gridY = static_cast<int>(gridPos.y) / gridVisualizer.getSquareSize();
                gridModel.setOccupied(gridX, gridY, true);

                // Sleep for a short duration to prevent rapid wall creation
                sf::sleep(sf::milliseconds(100));
            }
        }
    }


    void handleMouseButtonReleased(const sf::Event& event) {
        if (event.mouseButton.button == sf::Mouse::Left && isTracing) {
            end = window.mapPixelToCoords(sf::Mouse::getPosition(window));
            rays.emplace_back(start, end);
            isTracing = false;
        }
    }

    void handleKeyPressed(const sf::Event& event) {
        switch (event.key.code) {
            case sf::Keyboard::Escape:
                gridModel.reset();
                rays.clear();
                isTracing = false;
                break;
            case sf::Keyboard::R:
                gridModel.reset();
                break;
            // Add more cases as needed for other key functionalities
        }
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 800), "Grid World");
    int gridSize = 20;
    GridModel gridModel(gridSize);
    GridVisualizer gridVisualizer(window, gridModel, window.getSize().x / gridSize);
    std::vector<Ray> rays;
    sf::Vector2f start, end;
    bool isTracing = false;

    EventHandler eventHandler(window, gridModel, gridVisualizer, rays, start, end, isTracing);

    while (window.isOpen()) {
        eventHandler.processEvents();

        if (isTracing) {
            // Handle ray tracing and updating the model
            isTracing = false;
        }

        gridVisualizer.draw();

        window.display();
    }

    return 0;
}
