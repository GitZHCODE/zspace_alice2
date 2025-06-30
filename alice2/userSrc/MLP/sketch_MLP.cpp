// alice2 Base Sketch Template
// This is a template for creating user sketches in alice2

#define __MAIN__
#ifdef __MAIN__

#include <alice2.h>
#include <sketches/SketchRegistry.h>
#include <ML/polygonSDF_MLP.h>

using namespace alice2;

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
