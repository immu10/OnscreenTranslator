#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <ctime>

// Utility: get current local time as string
std::string current_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

int main() {
    // Initialize Tesseract
    tesseract::TessBaseAPI api;
    if (api.Init("tessdata", "eng")) {   // change this path!
        std::cerr << "Could not initialize tesseract.\n";
        return 1;
    }

    std::vector<std::string> images = {
        "img1.png",
        "img2.png",
        "img3.png"
    };

    for (const auto &filename : images) {
        std::cout << "[" << current_time() << "] Starting OCR for " << filename << "\n";

        auto start = std::chrono::high_resolution_clock::now();

        Pix *image = pixRead(filename.c_str());
        if (!image) {
            std::cerr << "Could not open " << filename << "\n";
            continue;
        }

        api.SetImage(image);
        char *outText = api.GetUTF8Text();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "[" << current_time() << "] Finished " << filename
                  << " — took " << duration.count() << " ms\n";
        std::cout << "Detected text:\n" << outText << "\n";

        delete[] outText;
        pixDestroy(&image);
    }

    api.End();
    return 0;
}
