#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include "../src/computeGeom/ComputeGraph.h"
#include <array>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>

using namespace alice2;

class HeGraphSketch : public ISketch {
private:
    std::shared_ptr<ComputeGraph> m_computeGraph;

    bool b_showVertices;
    bool b_showEdges;
    bool b_showHalfEdges;
    bool d_showNeighbors;

    int m_selectedVertexId;

public:
    HeGraphSketch()
        : b_showVertices(true)
        , b_showEdges(true)
        , b_showHalfEdges(false)
        , d_showNeighbors(false)
        , m_selectedVertexId(0) {
    }

    std::string getName() const override {
        return "Half-Edge Graph Demo";
    }

    std::string getDescription() const override {
        return "Visualizes graphs with half-edge connectivity queries";
    }

    void setup() override {
        scene().setBackgroundColor(Color(0.12f, 0.12f, 0.12f));
        scene().setShowGrid(true);
        scene().setGridSize(8.0f);
        scene().setGridDivisions(8);
        scene().setShowAxes(true);
        scene().setAxesLength(2.0f);

        createTestGraph();

        std::cout << "Half-Edge Graph Sketch initialized" << std::endl;
    }

    void update(float /*deltaTime*/) override {}

    void draw(Renderer& renderer, Camera& /*camera*/) override {
        if (!m_computeGraph) {
            return;
        }

        if (b_showHalfEdges) {
            drawHalfEdges(renderer);
        }

        if (d_showNeighbors) {
            drawVertexNeighbors(renderer);
        }

        drawUI(renderer);
    }

    bool onKeyPress(unsigned char key, int /*x*/, int /*y*/) override {
        if (!m_computeGraph) {
            return false;
        }

        switch (key) {
            case 'v':
                b_showVertices = !b_showVertices;
                m_computeGraph->setShowVertices(b_showVertices);
                return true;

            case 'e':
                b_showEdges = !b_showEdges;
                m_computeGraph->setShowEdges(b_showEdges);
                return true;

            case 'h':
                b_showHalfEdges = !b_showHalfEdges;
                return true;

            case 'n':
                d_showNeighbors = !d_showNeighbors;
                if (d_showNeighbors) {
                    logSelectedVertexInfo();
                }
                return true;

            case '1':
                cycleSelection(-1);
                return true;

            case '2':
                cycleSelection(1);
                return true;
        }
        return false;
    }

    void cleanup() override {}

    std::string getAuthor() const override {
        return "alice2 User";
    }

private:
    void cycleSelection(int delta) {
        if (!m_computeGraph) {
            return;
        }

        const int count = static_cast<int>(m_computeGraph->getVertices().size());
        if (count <= 0) {
            m_selectedVertexId = 0;
            return;
        }

        m_selectedVertexId = (m_selectedVertexId + delta) % count;
        if (m_selectedVertexId < 0) {
            m_selectedVertexId += count;
        }

        logSelectedVertexInfo();
    }

    void createTestGraph() {
        GraphData data;
        data.clear();

        // Cube vertices.
        const std::array<Vec3, 8> cubePositions = {
            Vec3(-1, -1, -1), // 0
            Vec3( 1, -1, -1), // 1
            Vec3( 1,  1, -1), // 2
            Vec3(-1,  1, -1), // 3
            Vec3(-1, -1,  1), // 4
            Vec3( 1, -1,  1), // 5
            Vec3( 1,  1,  1), // 6
            Vec3(-1,  1,  1)  // 7
        };

        const Color cubeColor(0.55f, 0.75f, 1.0f);
        for (const auto& pos : cubePositions) {
            data.addVertex(pos, cubeColor);
        }

        const std::array<std::pair<int, int>, 12> cubeEdges = {{
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7}
        }};

        for (const auto& [a, b] : cubeEdges) {
            if (data.addEdge(a, b) < 0) {
                std::cout << "Failed to add cube edge (" << a << ", " << b << ")" << std::endl;
            }
        }

        // N-gon ring vertices (offset above the cube for clarity).
        std::vector<int> ringIndices;
        const int ringCount = 8;
        ringIndices.reserve(ringCount);
        const float ringRadius = 2.0f;
        constexpr float twoPi = 6.28318530718f;
        const Color ringStart(1.0f, 0.55f, 0.3f);
        const Color ringEnd(0.3f, 0.95f, 0.8f);

        for (int i = 0; i < ringCount; ++i) {
            float t = static_cast<float>(i) / ringCount;
            float angle = t * twoPi;
            Vec3 position(ringRadius * std::cos(angle), ringRadius * std::sin(angle), (i % 2 == 0) ? 2.0f : 1.4f);
            Color vertexColor = Color::lerp(ringStart, ringEnd, static_cast<float>(i) / (ringCount - 1));
            ringIndices.push_back(data.addVertex(position, vertexColor));
        }

        for (int i = 0; i < ringCount; ++i) {
            int a = ringIndices[i];
            int b = ringIndices[(i + 1) % ringCount];
            if (data.addEdge(a, b) < 0) {
                std::cout << "Failed to add ring edge (" << a << ", " << b << ")" << std::endl;
            }
        }

        m_computeGraph = std::make_shared<ComputeGraph>("TestGraph", data);
        m_computeGraph->setShowVertices(b_showVertices);
        m_computeGraph->setShowEdges(b_showEdges);
        m_computeGraph->setVertexSize(8.0f);
        m_computeGraph->setEdgeWidth(2.5f);
        m_computeGraph->setEdgeColor(Color(0.7f, 0.7f, 0.9f));

        m_selectedVertexId = 0;
        logSelectedVertexInfo();

        scene().addObject(m_computeGraph);

        printGraphStatistics();
    }

    void logSelectedVertexInfo() const {
        if (!m_computeGraph) {
            return;
        }

        const auto& vertices = m_computeGraph->getVertices();
        if (vertices.empty() || m_selectedVertexId < 0 || m_selectedVertexId >= static_cast<int>(vertices.size())) {
            return;
        }

        auto vertex = m_computeGraph->getVertex(m_selectedVertexId);
        if (!vertex) {
            return;
        }

        auto neighbors = vertex->getNeighbors();
        std::cout << "Selected vertex " << vertex->getId() << " has " << neighbors.size() << " neighbors:";
        for (const auto& neighbor : neighbors) {
            if (neighbor) {
                std::cout << ' ' << neighbor->getId();
            }
        }
        std::cout << std::endl;
    }

    void printGraphStatistics() {
        if (!m_computeGraph) {
            return;
        }

        const auto& heData = m_computeGraph->getHeGraphData();
        std::cout << "\n=== Half-Edge Graph Statistics ===" << std::endl;
        std::cout << "Vertices: " << heData.vertices.size() << std::endl;
        std::cout << "Half-edges: " << heData.halfedges.size() << std::endl;
        std::cout << "Edges: " << heData.edges.size() << std::endl;
    }

    void drawHalfEdges(Renderer& renderer) {
        const auto& heData = m_computeGraph->getHeGraphData();
        for (const auto& halfedge : heData.halfedges) {
            auto start = halfedge->getStartVertex();
            auto end = halfedge->getVertex();
            if (!start || !end) {
                continue;
            }

            Vec3 startPos = start->getPosition();
            Vec3 endPos = end->getPosition();
            Vec3 direction = (endPos - startPos).normalized();
            Vec3 offsetEnd = startPos + direction * (endPos - startPos).length() * 0.85f;

            Color color = Color(0.2f, 1.0f, 0.6f);
            renderer.drawLine(startPos, offsetEnd, color, 2.0f);
            renderer.drawPoint(offsetEnd, color, 5.0f);
        }
    }

    void drawVertexNeighbors(Renderer& renderer) {
        if (!m_computeGraph) {
            return;
        }
        const auto& vertices = m_computeGraph->getVertices();
        if (vertices.empty()) {
            return;
        }
        if (m_selectedVertexId < 0 || m_selectedVertexId >= static_cast<int>(vertices.size())) {
            return;
        }

        auto vertex = m_computeGraph->getVertex(m_selectedVertexId);
        if (!vertex) {
            return;
        }

        renderer.drawPoint(vertex->getPosition(), Color(1.0f, 1.0f, 0.2f), 10.0f);

        auto neighbors = vertex->getNeighbors();
        for (const auto& neighbor : neighbors) {
            if (!neighbor) {
                continue;
            }
            Color edgeColor(1.0f, 0.4f, 0.2f);
            renderer.drawLine(vertex->getPosition(), neighbor->getPosition(), edgeColor, 3.0f);
            renderer.drawPoint(neighbor->getPosition(), Color(1.0f, 0.6f, 0.3f), 8.0f);
        }
    }

    void drawUI(Renderer& renderer) {
        renderer.setColor(Color(1.0f, 1.0f, 1.0f));
        renderer.drawString(getName(), 10, 30);
        renderer.drawString(getDescription(), 10, 50);

        renderer.setColor(Color(0.0f, 1.0f, 1.0f));
        renderer.drawString("FPS: " + std::to_string(Application::getInstance()->getFPS()), 10, 70);

        renderer.setColor(Color(0.7f, 0.7f, 0.7f));
        renderer.drawString("Controls:", 10, 110);
        renderer.drawString("'v' - Toggle vertices", 10, 130);
        renderer.drawString("'e' - Toggle edges", 10, 150);
        renderer.drawString("'h' - Toggle half-edges", 10, 170);
        renderer.drawString("'n' - Toggle neighbor highlight", 10, 190);
        renderer.drawString("'1/2' - Cycle selected vertex (" + std::to_string(m_selectedVertexId) + ")", 10, 210);

        if (m_computeGraph) {
            const auto& heData = m_computeGraph->getHeGraphData();
            renderer.setColor(Color(0.9f, 0.9f, 0.5f));
            renderer.drawString("Graph Info:", 10, 250);
            renderer.drawString("Vertices: " + std::to_string(heData.vertices.size()), 10, 270);
            renderer.drawString("Half-edges: " + std::to_string(heData.halfedges.size()), 10, 290);
            renderer.drawString("Edges: " + std::to_string(heData.edges.size()), 10, 310);
        }
    }
        };

ALICE2_REGISTER_SKETCH_AUTO(HeGraphSketch)

#endif // __MAIN__
