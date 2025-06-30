// alice2 Base Sketch Template
// This is a template for creating user sketches in alice2

#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <ML/genericMLP.h>
#include <computeGeom/scalarField.h>
#include <fstream>
#include <charconv>
#include <limits>
#include <cmath>

using namespace alice2;

class PolygonSDF_MLP : public MLP
{
public:
    using MLP::MLP;

    std::vector<Vec3> polygon;
    std::vector<Vec3> trainingSamples;
    std::vector<float> sdfGT;
    std::vector<float> losses;
    std::vector<float> losses_ang;

    std::vector<Vec3> fittedCenters;
    std::vector<float> fittedRadii;

    int number_sdf;
    double radius = 8.;
    float smoothK = 3.0f;
    Vec3 sunDir = Vec3(1, 1, 0);

    ScalarField2D generatedField;
    int epoch = 0;

    void GenerateField(std::vector<float> &x)
    {
        auto out = forward(x);

        std::vector<Vec3> centers(number_sdf);
        std::vector<float> radii(number_sdf);

        decodeOutput(out, centers, radii);

        GenerateField(centers, radii);
    }

    void GenerateField(std::vector<Vec3> &centers, std::vector<float> &radii)
    {
        // for (auto& r : radii)r = radius;
        generatedField.clearField();
        //  generatedField.addVoronoi(trainingSamples);

        std::pair<int, int> RES = generatedField.get_resolution();
        std::vector<float> fieldValues = generatedField.get_values();
        std::vector<Vec3> fieldPoints = generatedField.get_points();

        for (size_t i = 0; i < fieldValues.size(); ++i)
            fieldValues[i] = blendOrientedBoxSDFs(fieldPoints[i], centers, radii);

        rescaleToRange(fieldValues);
        generatedField.set_values(fieldValues);
        //generatedField.normalise();
    }

void rescaleToRange(std::vector<float>& values,
                    float targetMin = -1.0f,
                    float targetMax =  1.0f)
{
    if (values.empty()) return;

    // find per‐sign extrema
    float minVal[2] = {
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
    };
    float maxVal[2] = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    };

    for (float v : values) {
        int idx = (v >= 0.0f ? 0 : 1);
        minVal[idx] = std::min(minVal[idx], v);
        maxVal[idx] = std::max(maxVal[idx], v);
    }

    // avoid zero‐division
    float range[2] = {
        std::max(maxVal[0] - minVal[0], 1e-6f),
        std::max(maxVal[1] - minVal[1], 1e-6f)
    };

    // remap each value:
    //  - positives  → [0, targetMax]
    //  - negatives  → [targetMin, 0]
    for (float& v : values) {
        if (v >= 0.0f) {
            float t = (v - minVal[0]) / range[0];          // in [0,1]
            v = std::lerp(0.0f,      targetMax, t);         // → [0, targetMax]
        }
        else {
            float t = (v - minVal[1]) / range[1];          // in [0,1]
            v = std::lerp(targetMin, 0.0f,      t);         // → [targetMin, 0]
        }
    }
}

    float blendOrientedBoxSDFs(Vec3 pt, std::vector<Vec3> &centers, std::vector<float> &angles, float width = 8.0f, float height = 6.0f, float k = 3.0f)
    {
        float d = 1e6;
        for (int i = 0; i < centers.size(); i++)
        {
            float dist = orientedBoxSDF(pt, centers[i], width, height, angles[i]);
            d = std::min(d, dist);
            ; // smin(d, dist, k);
        }
        return d;
    }

    float orientedBoxSDF(Vec3 pt, Vec3 center, float width, float height, float angleRad)
    {
        Vec3 d = pt - center;

        float cosA = cos(angleRad);
        float sinA = sin(angleRad);

        float localX = d.x * cosA + d.y * sinA;
        float localY = -d.x * sinA + d.y * cosA;

        float dx = fabs(localX) - width * 0.5f;
        float dy = fabs(localY) - height * 0.5f;

        float ax = std::max(dx, 0.0f);
        float ay = std::max(dy, 0.0f);

        float insideDist = std::min(std::max(dx, dy), 0.0f);
        return sqrtf(ax * ax + ay * ay) + insideDist;
    }

    void decodeOutput(const std::vector<float> &out, std::vector<Vec3> &centers, std::vector<float> &angles)
    {
        centers.resize(number_sdf);
        angles.resize(number_sdf);
        for (int i = 0; i < number_sdf; i++)
        {
            int idx = i * 4;
            centers[i] = Vec3(out[idx + 0], out[idx + 1], 0);

            Vec3 dir(out[idx + 2], out[idx + 3], 0);
            //   dir = gradientAt_polygonSDF(centers[i], polygon);

            dir.normalize();
            // dir = dir ^ Vec3(0, 0, 1);

            angles[i] = atan2(dir.y, dir.x);
        }
    }

    float evaluateLoss(std::vector<Vec3> &centers, std::vector<float> &angles)
    {
        const int N = trainingSamples.size();
        const int numLossTypes = 2; // 0: coverage, 1: angular (add more as needed)

        std::vector<std::vector<float>> lossesByType(numLossTypes, std::vector<float>(N, 0.0f));

        Vec3 sunDir(1, 1, 0);
        sunDir.normalize();

        // Step 1: compute raw losses
        for (int i = 0; i < N; i++)
        {
            Vec3 pt = trainingSamples[i];

            // Loss 0: coverage (MSE)
            float pred = blendCircleSDFs(pt, centers, angles, smoothK);
            float err = pred - sdfGT[i];
            lossesByType[0][i] = err * err;

            // Loss 1: angular alignment (squared angle)
            Vec3 grad = gradientAt(pt, centers, angles);            // gradient of blendedSDF
            Vec3 grad_polygon = gradientAt_polygonSDF(pt, polygon); // gradient of polygonSDF;
            grad.normalize();
            grad = grad.cross(Vec3(0, 0, 1));
            grad_polygon.normalize();

            float angleErr = angleBetween(grad, sunDir);
            lossesByType[1][i] = angleErr * angleErr;
        }

        // Step 2: normalize each loss type to [0,1]
        std::vector<bool> normalizeLoss = {false, true}; // match number of loss types

        for (int t = 0; t < numLossTypes; t++)
        {
            if (!normalizeLoss[t])
                continue;

            float minVal = 1e6f, maxVal = -1e6f;
            for (float v : lossesByType[t])
            {
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }

            float range = std::max(maxVal - minVal, 1e-6f);
            for (float &v : lossesByType[t])
            {
                v = (v - minVal) / range;
            }
        }

        // Step 3: weighted sum of all loss types
        std::vector<float> weights = {1, 1}; // must match numLossTypes
        float totalLoss = 0.0f;
        for (int i = 0; i < N; i++)
        {
            float combined = 0.0f;
            for (int t = 0; t < numLossTypes; t++)
            {
                combined += weights[t] * lossesByType[t][i];
            }
            totalLoss += combined;
        }

        // Optional debug access: you may assign lossesByType[0] → `losses`, lossesByType[1] → `losses_ang`
        losses = lossesByType[0];
        losses_ang = lossesByType[1];

        return totalLoss / trainingSamples.size();
    }

    float computeLoss(std::vector<float> &x, std::vector<float> &dummy) override
    {
        auto out = forward(x);
        std::vector<Vec3> centers;
        std::vector<float> angles;
        decodeOutput(out, centers, angles);

        epoch++;
        return evaluateLoss(centers, angles);
    }

    float blendCircleSDFs(Vec3 pt, std::vector<Vec3> &centers, std::vector<float> &radii, float k)
    {
        float d = 1e6;
        for (int i = 0; i < centers.size(); i++)
        {
            float dist = pt.distanceTo(centers[i]) - radii[i];
            d = ScalarFieldUtils::smooth_min(d, dist, k);
        }
        return d;
    }

    Vec3 gradientAt(Vec3 pt, std::vector<Vec3> &centers, std::vector<float> &angles, float h = 0.1f)
    {
        float dx = blendOrientedBoxSDFs(pt + Vec3(h, 0, 0), centers, angles) -
                   blendOrientedBoxSDFs(pt - Vec3(h, 0, 0), centers, angles);

        float dy = blendOrientedBoxSDFs(pt + Vec3(0, h, 0), centers, angles) -
                   blendOrientedBoxSDFs(pt - Vec3(0, h, 0), centers, angles);

        Vec3 ret(dx, dy, 0);
        ret.normalize();
        return ret;
    }

    Vec3 gradientAt_polygonSDF(const Vec3 &pt, std::vector<Vec3> &polygon, float h = 0.1f)
    {
        float dx = polygonSDF(pt + Vec3(h, 0, 0), polygon) -
                   polygonSDF(pt - Vec3(h, 0, 0), polygon);

        float dy = polygonSDF(pt + Vec3(0, h, 0), polygon) -
                   polygonSDF(pt - Vec3(0, h, 0), polygon);

        Vec3 ret(dx, dy, 0);
        ret.normalize();
        return ret;
    }
    float angleBetween(Vec3 &a, Vec3 &b)
    {
        float dot = a.x * b.x + a.y * b.y;
        float det = a.x * b.y - a.y * b.x;
        return atan2(det, dot); // angle in radians
    }

    void computeGradient(std::vector<float> &x, std::vector<float> &dummy, std::vector<float> &gradOut) override
    {
        auto out = forward(x);
        float eps = 0.01f;

        std::vector<Vec3> baseCenters;
        std::vector<float> baseAngles;
        decodeOutput(out, baseCenters, baseAngles);

        float baseLoss = evaluateLoss(baseCenters, baseAngles);

        gradOut.assign(out.size(), 0.0f);

        for (int i = 0; i < out.size(); ++i)
        {
            std::vector<float> outPerturbed = out;
            outPerturbed[i] += eps;

            std::vector<Vec3> centers;
            std::vector<float> angles;
            decodeOutput(outPerturbed, centers, angles);

            float lossPerturbed = evaluateLoss(centers, angles);
            gradOut[i] = (lossPerturbed - baseLoss) / eps;
        }
    }

    bool isInsidePolygon(const Vec3 &p, std::vector<Vec3> &poly)
    {
        int windingNumber = 0;

        for (int i = 0; i < poly.size(); i++)
        {
            Vec3 &a = poly[i];
            Vec3 &b = poly[(i + 1) % poly.size()];

            if (a.y <= p.y)
            {
                if (b.y > p.y && ((b - a).cross(p - a)).z > 0)
                    ++windingNumber;
            }
            else
            {
                if (b.y <= p.y && ((b - a).cross(p - a)).z < 0)
                    --windingNumber;
            }
        }

        return (windingNumber != 0);
    }

    float polygonSDF(const Vec3 &p, std::vector<Vec3> &poly)
    {
        float minDist = 1e6;
        int n = poly.size();

        for (int i = 0; i < n; i++)
        {
            Vec3 a = poly[i];
            Vec3 b = poly[(i + 1) % n];

            Vec3 ab = b - a;
            Vec3 ap = p - a;

            float t = std::max(0.0f, std::min(1.0f, (ab.dot(ap)) / (ab.dot(ab))));
            Vec3 proj = a + ab * t;
            float d = p.distanceTo(proj);
            minDist = std::min(minDist, d);
        }

        return minDist * (isInsidePolygon(p, poly) ? -1.0f : 1.0f);
    }

    // Utility functions

    void samplePoints(std::vector<Vec3> &trainingSamples, std::vector<float> &sdfGT, std::vector<Vec3> &polygon)
    {
        // collect input-output pairs of information

        trainingSamples.clear();
        sdfGT.clear();

        for (float x = -50; x <= 50; x += 5.0f)
        {
            for (float y = -50; y <= 50; y += 5.0f)
            {
                Vec3 pt(x, y, 0);
                if (isInsidePolygon(pt, polygon))
                {
                    trainingSamples.push_back(pt);            // input exmaples
                    sdfGT.push_back(polygonSDF(pt, polygon)); // known output expected for the input
                }
            }
        }

        std::cout << "Training samples: " << trainingSamples.size() << std::endl;
    }

    // Visualization methods using alice2 renderer

    void drawLossText(Renderer& renderer, float startY = 150)
    {
        if (losses.empty() || losses_ang.empty()) return;

        char s[100];
        float lossSum = 0;
        float loss_A_Sum = 0;

        for (int i = 0; i < losses_ang.size(); i++)
        {
            lossSum += losses[i];
            loss_A_Sum += losses_ang[i];
        }

        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
        sprintf(s, " loss %1.2f", lossSum / trainingSamples.size());
        renderer.drawString(std::string(s), 10, startY);

        sprintf(s, " loss_ang %1.2f", loss_A_Sum);
        renderer.drawString(std::string(s), 10, startY + 15);
    }

    void drawLossBarGraph(Renderer& renderer, const std::vector<float>& losses, float startPtX, float startPtY, float screenWidth = 800, float barHeight = 50)
    {
        if (losses.empty()) return;

        int N = losses.size();
        float barSpacing = screenWidth / (float)N;

        // Normalize losses to [0, 1]
        float minVal = 1e6f, maxVal = -1e6f;
        for (float v : losses)
        {
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
        float range = std::max(maxVal - minVal, 1e-6f);  // avoid divide by zero

        for (int i = 0; i < N; i++)
        {
            float normalized = (losses[i] - minVal) / range;
            float x = startPtX + i * barSpacing;
            float h = barHeight * normalized;

            float r, g, b;
            ScalarFieldUtils::get_jet_color(normalized * 2.0f - 1.0f, r, g, b);

            Vec2 start(x, startPtY);
            Vec2 end(x, startPtY + h);
            renderer.draw2dLine(start, end, Vec3(r, g, b));
        }
    }

    void visualiseField(Renderer& renderer, float threshold = 0.01, bool drawField = true)
    {
        if (drawField) generatedField.draw_points(renderer,1);
        generatedField.draw_values(renderer);
    }

    void visualiseGradients(Renderer& renderer, std::vector<float>& x)
    {
        auto out = forward(x);

        std::vector<Vec3> centers(number_sdf);
        std::vector<float> angles(number_sdf);

        decodeOutput(out, centers, angles);

        // Draw gradients for SDF centers
        for (int i = 0; i < number_sdf; i++)
        {
            Vec3 grad_polygon = gradientAt_polygonSDF(centers[i], polygon);
            grad_polygon.normalize();

            Vec3 a = centers[i];  // Use Vec3 directly instead of Alice::vec

            renderer.drawLine(a, a + grad_polygon * 3.0f, Vec3(0.0f, 0.0f, 0.0f));

            // Local coordinate system visualization
            float cosA = cos(angles[i]);
            float sinA = sin(angles[i]);

            Vec3 axisX(cosA, -sinA, 0); // local X direction
            Vec3 axisY(sinA, cosA, 0);  // local Y direction

            Vec3 grad = axisY; // Use local Y axis as gradient direction
            grad.normalize();

            renderer.drawLine(a, a + grad * 4.0f, Vec3(1.0f, 0.0f, 0.0f));
        }

        // Draw gradients for training samples
        for (int i = 0; i < trainingSamples.size(); i++)
        {
            Vec3 a = trainingSamples[i];

            Vec3 grad_polygon = gradientAt_polygonSDF(trainingSamples[i], polygon);
            grad_polygon.normalize();

            renderer.drawLine(a, a + grad_polygon, Vec3(0.0f, 0.0f, 0.0f));

            Vec3 grad = gradientAt(trainingSamples[i], centers, angles);
            grad.normalize();

            renderer.drawLine(a, a + grad, Vec3(1.0f, 0.0f, 0.0f));
        }
    }

    // Helper method to draw a circle using line segments
    void drawCircle(Renderer& renderer, const Vec3& center, float radius, int segments, const Vec3& color)
    {
        renderer.setColor(color);
        const float PI = 3.14159265359f;
        for (int i = 0; i < segments; i++)
        {
            float angle1 = (float)i / segments * 2.0f * PI;
            float angle2 = (float)(i + 1) / segments * 2.0f * PI;

            Vec3 p1 = center + Vec3(cos(angle1) * radius, sin(angle1) * radius, 0);
            Vec3 p2 = center + Vec3(cos(angle2) * radius, sin(angle2) * radius, 0);

            renderer.drawLine(p1, p2);
        }
    }
};

#define NUM_CENTERS 8

class MLPSketch : public ISketch
{
public:
    MLPSketch() = default;
    ~MLPSketch() = default;

    bool run = false;
    double lr = 0.1;
    double tv = -0.005;

    // Visualization flags
    bool showGradients = true;
    bool showLossGraph = true;
    bool showField = true;
    bool showCenters = true;
    bool showTrainingSamples = true;
    bool showPolygon = true;
    bool showMLPViz = true;

    PolygonSDF_MLP mlp;
    std::vector<float> grads;
    std::vector<float> mlp_input_data;
    std::vector<Vec3> polygon;
    std::vector<Vec3> trainingSamples;
    std::vector<float> sdfGT;

    ContourData contours;

    // Sketch information
    std::string getName() const override
    {
        return "Base Sketch";
    }

    std::string getDescription() const override
    {
        return "A basic template sketch for alice2";
    }

    std::string getAuthor() const override
    {
        return "alice2 User";
    }

    // Sketch lifecycle
    void setup() override
    {
        // Initialize your sketch here
        // This is called once when the sketch is loaded

        // Example: Set background color
        scene().setBackgroundColor(Vec3(0.15f, 0.15f, 0.15f));
        std::cout << "Background color set to light gray" << std::endl;

        // Example: Enable grid
        scene().setShowGrid(false);
        scene().setGridSize(10.0f);
        scene().setGridDivisions(10);

        // Example: Enable axes
        scene().setShowAxes(false);
        scene().setAxesLength(2.0f);

        initializeMLP(); // create MLP

        // load boudnary polygon from a text file;
        loadPolygonFromCSV("C://Users//taizhong_chen//source//repos//GitZHCODE//zspace_alice2//alice2//data//polygon__.txt", polygon);

        // calculate training set
        mlp.samplePoints(trainingSamples, sdfGT, polygon);

        mlp.polygon = polygon;
        mlp.trainingSamples = trainingSamples;
        mlp.sdfGT = sdfGT;
        mlp.losses.resize(sdfGT.size());
        mlp.losses_ang.resize(sdfGT.size());
        //

        std::pair<int, int> RES = mlp.generatedField.get_resolution();
        std::vector<float> fieldValues = mlp.generatedField.get_values();
        std::vector<Vec3> fieldPoints = mlp.generatedField.get_points();

        for (size_t i = 0; i < fieldValues.size(); ++i)
            fieldValues[i] = mlp.polygonSDF(fieldPoints[i], polygon);

        mlp.rescaleToRange(fieldValues);
        mlp.generatedField.set_values(fieldValues);
        //mlp.generatedField.normalise();
    }

    void update(float deltaTime) override
    {
        // Update your sketch logic here
        // This is called every frame
        // deltaTime is the time elapsed since the last frame in seconds

        if (run)
        {
            grads.clear();
            std::vector<float> dummy;
            float loss = mlp.computeLoss(mlp_input_data, dummy);
            mlp.computeGradient(mlp_input_data, dummy, grads);

            mlp.backward(grads, lr);

            std::cout << "loss :" << loss << std::endl;
            lr *= 0.99;

            mlp.GenerateField(mlp_input_data);

            contours = mlp.generatedField.get_contours(0.02f);

        }
    }

    void draw(Renderer &renderer, Camera &camera) override
    {
        // Draw your custom content here

        for(auto& line : contours.line_segments)
        renderer.drawLine(line.first,line.second, Vec3(1.0f, 1.0f, 1.0f), 2.0f);

        // This is called every frame after update()

        // Draw the generated scalar field
        if (showField) {
            mlp.visualiseField(renderer, 0.01f, true);
        }

        // Draw fitted centers as red circles
        if (showCenters) {
            for (auto& c : mlp.fittedCenters) {
                mlp.drawCircle(renderer, c, 3.0f, 32, Vec3(1.0f, 0.0f, 0.0f));
            }
        }

        // Draw training samples as blue points
        if (showTrainingSamples) {
            renderer.setColor(Vec3(0.0f, 0.0f, 1.0f));
            for (auto& p : trainingSamples) {
                renderer.drawPoint(p, Vec3(0.0f, 0.0f, 1.0f), 3.0f);
            }
        }

        // Draw polygon as black lines
        if (showPolygon && !polygon.empty()) {
            renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
            for (int i = 0; i < polygon.size(); i++) {
                int j = (i + 1) % polygon.size();
                renderer.drawLine(polygon[i], polygon[j]);
            }
        }

        // Draw gradients if we have MLP input data
        if (showGradients && !mlp_input_data.empty()) {
            mlp.visualiseGradients(renderer, mlp_input_data);
        }

        // Draw MLP network visualization (if available)
        if (showMLPViz) {
            // Note: MLP visualization would go here when implemented
            mlp.visualize(renderer, camera, Vec3(10, 450, 0), 200, 250);
        }

        // Draw loss text
        mlp.drawLossText(renderer, 460);

        // Draw loss bar graphs if we have loss data
        if (showLossGraph) {
            if (!mlp.losses.empty()) {
                mlp.drawLossBarGraph(renderer, mlp.losses, 10, 700, 200, 40);
            }
            if (!mlp.losses_ang.empty()) {
                mlp.drawLossBarGraph(renderer, mlp.losses_ang, 10, 750, 200, 40);
            }
        }

        // 2D text rendering (screen overlay)
        renderer.setColor(Vec3(1.0f, 1.0f, 1.0f));
        renderer.drawString(getName(), 10, 30);
        renderer.drawString(getDescription(), 10, 50);

        renderer.setColor(Vec3(0.0f, 1.0f, 1.0f));
        renderer.drawString("FPS: " + std::to_string((Application::getInstance()->getFPS())), 10, 70);

        // renderer.setColor(Vec3(0.75f, 0.75f, 0.75f));
        // renderer.drawString("'ESC' - Exit ", 10, 200);
        // renderer.drawString("'F'   - Extend view ", 10, 220);
        // renderer.drawString("'N'   - Switch to the next sketch ", 10, 240);
        // renderer.drawString("'P'   - Switch to the previous sketch ", 10, 260);
        // renderer.drawString("'U'   - Update field ", 10, 280);
        // renderer.drawString("'R'   - Toggle training ", 10, 300);
        // renderer.drawString("'G'   - Toggle gradients ", 10, 320);
        // renderer.drawString("'L'   - Toggle loss graph ", 10, 340);
        // renderer.drawString("'V'   - Toggle field visualization ", 10, 360);
        // renderer.drawString("'C'   - Toggle centers ", 10, 380);
        // renderer.drawString("'T'   - Toggle training samples ", 10, 400);
        // renderer.drawString("'O'   - Toggle polygon ", 10, 420);
        // renderer.drawString("'M'   - Toggle MLP visualization ", 10, 440);
    }

    void cleanup() override
    {
        // Clean up resources here
        // This is called when the sketch is unloaded
    }

    // Input handling (optional)
    bool onKeyPress(unsigned char key, int x, int y) override
    {
        // Handle keyboard input
        switch (key)
        {
        case 27: // ESC key
            // Example: Exit application
            return false; // Not handled - allow default exit
        case 'u':
        case 'U':
            mlp.GenerateField(mlp_input_data);
            return true;
        case 'r':
        case 'R':
            run = !run;
            return true;
        case 'g':
        case 'G':
            showGradients = !showGradients;
            return true;
        case 'l':
        case 'L':
            showLossGraph = !showLossGraph;
            return true;
        case 'v':
        case 'V':
            showField = !showField;
            return true;
        case 'c':
        case 'C':
            showCenters = !showCenters;
            return true;
        case 't':
        case 'T':
            showTrainingSamples = !showTrainingSamples;
            return true;
        case 'o':
        case 'O':
            showPolygon = !showPolygon;
            return true;
        case 'm':
        case 'M':
            showMLPViz = !showMLPViz;
            return true;
        }
        return false; // Not handled
    }

    bool onMousePress(int button, int state, int x, int y) override
    {
        // Handle mouse button input
        // button: 0=left, 1=middle, 2=right
        // state: 0=down, 1=up
        return false; // Not handled - allow default camera controls
    }

    bool onMouseMove(int x, int y) override
    {
        // Handle mouse movement
        return false; // Not handled - allow default camera controls
    }

    void initializeMLP()
    {
        int input_dim = NUM_CENTERS * 4;
        int output_dim = NUM_CENTERS * 4;
        std::vector<int> hidden_dims = {16};

        mlp = PolygonSDF_MLP(input_dim, hidden_dims, output_dim); // assumes MLP constructor initializes weights/biases
        mlp_input_data.assign(input_dim, 1.0f);                   // or use 0.0f for strict zero-input
        mlp.number_sdf = NUM_CENTERS;
    }

    void loadPolygonFromCSV(const std::string &filename, std::vector<Vec3> &polygon)
    {
        polygon.clear();
        std::ifstream file(filename);
        std::string line;

        while (std::getline(file, line))
        {
            std::string_view sv{line};
            auto comma = sv.find(',');
            float x{}, y{};
            std::from_chars(sv.data(), sv.data() + comma, x);
            std::from_chars(sv.data() + comma + 1, sv.data() + sv.size(), y);
            polygon.emplace_back(x, y, 0.0f);
        }
        std::cout << polygon.size() << " polygon size" << std::endl;
    }
};
// Register the sketch with alice2 (both old and new systems)
ALICE2_REGISTER_SKETCH(MLPSketch)
ALICE2_REGISTER_SKETCH_AUTO(MLPSketch)

#endif // __MAIN__
