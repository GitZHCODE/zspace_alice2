#include "zSpaceDraw.h"

#if ALICE2_WITH_ZSPACE_CORE

#include "../core/Renderer.h"
#include <zspace/interface.h>

#include <string>
#include <vector>

namespace alice2 {
namespace {

    Vec3 toVec3(const zSpace::zVector& p)
    {
        return Vec3(p.x, p.y, p.z);
    }

    Color toColor(const zSpace::zColor& c)
    {
        return Color(c.r, c.g, c.b, c.a);
    }

    void appendTriangle(std::vector<Vec3>& vertices, const Vec3& a, const Vec3& b, const Vec3& c)
    {
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
    }

    void appendTriangle(std::vector<Color>& colors, const Color& a, const Color& b, const Color& c)
    {
        colors.push_back(a);
        colors.push_back(b);
        colors.push_back(c);
    }

} // namespace

    void drawZSpaceMesh(Renderer& renderer, zSpace::zObjectMesh& mesh, const zDisplayMeshSetting& display)
    {
        if (display.showFaces) {
            std::vector<Vec3> triangles;
            std::vector<Color> vertexColors;
            zSpace::zColorArray meshVertexColors;

            if (display.useVertexColors) {
                zSpace::zFnMesh fn(mesh);
                fn.getVertexColors(meshVertexColors);
            }

            for (zSpace::zItMeshFace face(mesh); !face.end(); face++) {
                zSpace::zIntArray vertexIds;
                face.getVertices(vertexIds);
                if (vertexIds.size() < 3) continue;

                zSpace::zPointArray facePositions;
                face.getVertexPositions(facePositions);
                if (facePositions.size() < 3) continue;

                const Vec3 root = toVec3(facePositions[0]);
                for (size_t i = 1; i + 1 < facePositions.size(); ++i) {
                    appendTriangle(triangles, root, toVec3(facePositions[i]), toVec3(facePositions[i + 1]));

                    if (display.useVertexColors &&
                        vertexIds[0] >= 0 && vertexIds[i] >= 0 && vertexIds[i + 1] >= 0 &&
                        vertexIds[0] < static_cast<int>(meshVertexColors.size()) &&
                        vertexIds[i] < static_cast<int>(meshVertexColors.size()) &&
                        vertexIds[i + 1] < static_cast<int>(meshVertexColors.size())) {
                        appendTriangle(vertexColors,
                                       toColor(meshVertexColors[vertexIds[0]]),
                                       toColor(meshVertexColors[vertexIds[i]]),
                                       toColor(meshVertexColors[vertexIds[i + 1]]));
                    } else if (display.useVertexColors) {
                        appendTriangle(vertexColors, display.faceColor, display.faceColor, display.faceColor);
                    }
                }
            }

            if (!triangles.empty()) {
                std::vector<Color> colors;
                const Color* colorData = nullptr;
                if (display.useVertexColors && vertexColors.size() == triangles.size()) {
                    colorData = vertexColors.data();
                } else {
                    colors.assign(triangles.size(), display.faceColor);
                    colorData = colors.data();
                }
                renderer.drawMesh(triangles.data(), nullptr, colorData, static_cast<int>(triangles.size()), nullptr, 0, false);
            }
        }

        if (display.showEdges) {
            for (zSpace::zItMeshEdge edge(mesh); !edge.end(); edge++) {
                zSpace::zPointArray edgePositions;
                edge.getVertexPositions(edgePositions);
                if (edgePositions.size() == 2) {
                    renderer.drawLine(toVec3(edgePositions[0]), toVec3(edgePositions[1]), display.edgeColor, display.edgeWidth);
                }
            }
        }

        if (display.showVertices) {
            for (zSpace::zItMeshVertex vertex(mesh); !vertex.end(); vertex++) {
                renderer.drawPoint(toVec3(vertex.getPosition()), display.vertexColor, display.vertexSize);
            }
        }
    }

    void drawZSpaceGraph(Renderer& renderer, zSpace::zObjectGraph& graph, const zDisplayGraphSetting& display)
    {
        if (display.showEdges) {
            for (zSpace::zItGraphEdge edge(graph); !edge.end(); edge++) {
                zSpace::zPointArray edgePositions;
                edge.getVertexPositions(edgePositions);
                if (edgePositions.size() == 2) {
                    renderer.drawLine(toVec3(edgePositions[0]), toVec3(edgePositions[1]), display.edgeColor, display.edgeWidth);
                }
            }
        }

        if (display.showVertices) {
            for (zSpace::zItGraphVertex vertex(graph); !vertex.end(); vertex++) {
                renderer.drawPoint(toVec3(vertex.getPosition()), display.vertexColor, display.vertexSize);
            }
        }

        if (display.drawVertexIds) {
            renderer.setColor(display.vertexIdColor);
            for (zSpace::zItGraphVertex vertex(graph); !vertex.end(); vertex++) {
                renderer.drawText(std::to_string(vertex.getId()), toVec3(vertex.getPosition()), display.vertexIdSize);
            }
        }

        if (display.drawEdgeIds) {
            renderer.setColor(display.edgeIdColor);
            for (zSpace::zItGraphEdge edge(graph); !edge.end(); edge++) {
                zSpace::zPointArray edgePositions;
                edge.getVertexPositions(edgePositions);
                if (edgePositions.size() == 2) {
                    Vec3 center = (toVec3(edgePositions[0]) + toVec3(edgePositions[1])) * 0.5f;
                    renderer.drawText(std::to_string(edge.getId()), center, display.edgeIdSize);
                }
            }
        }
    }

    void drawZSpacePointCloud(Renderer& renderer, zSpace::zObjectPointCloud& points, const zDisplayPointCloudSetting& display)
    {
        for (zSpace::zItPointCloudVertex vertex(points); !vertex.end(); vertex++) {
            renderer.drawPoint(toVec3(vertex.getPosition()), display.vertexColor, display.vertexSize);
        }
    }

} // namespace alice2

#endif // ALICE2_WITH_ZSPACE_CORE
