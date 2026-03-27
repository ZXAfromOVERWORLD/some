import { useState } from "react";
import { MapPin, Upload, Send, Navigation2 } from "lucide-react";

export default function ComplaintPage() {
  const [location, setLocation] = useState("");
  const [description, setDescription] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [markerPosition, setMarkerPosition] = useState<{ x: number; y: number } | null>(null);

  const handleMapClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setMarkerPosition({ x, y });
    setLocation(`Lat: ${y.toFixed(2)}, Lng: ${x.toFixed(2)}`);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAutoDetect = () => {
    const randomX = Math.random() * 100;
    const randomY = Math.random() * 100;
    setMarkerPosition({ x: randomX, y: randomY });
    setLocation(`Lat: ${randomY.toFixed(2)}, Lng: ${randomX.toFixed(2)} (Auto-detected)`);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert("Complaint submitted successfully!");
    setLocation("");
    setDescription("");
    setSelectedImage(null);
    setMarkerPosition(null);
  };

  return (
    <div className="h-[calc(100vh-64px)] flex">
      {/* Left Side - Form */}
      <div className="w-1/2 bg-white p-8 overflow-y-auto border-r border-border">
        <div className="max-w-xl">
          <h2 className="text-2xl font-semibold text-foreground mb-2">Submit a Complaint</h2>
          <p className="text-muted-foreground mb-8">Report waste management issues in your area</p>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Location Input */}
            <div>
              <label className="block text-foreground mb-2">Location</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Enter location or click on map"
                  className="flex-1 px-4 py-3 bg-input-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2ECC71]"
                  required
                />
                <button
                  type="button"
                  onClick={handleAutoDetect}
                  className="px-4 py-3 bg-secondary text-foreground rounded-lg hover:bg-secondary/80 transition-colors flex items-center gap-2 whitespace-nowrap"
                >
                  <Navigation2 className="h-4 w-4" />
                  Auto-detect
                </button>
              </div>
              <p className="text-xs text-muted-foreground mt-2">Click on the map to select location</p>
            </div>

            {/* Image Upload */}
            <div>
              <label className="block text-foreground mb-2">Upload Image</label>
              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-[#2ECC71] transition-colors">
                {selectedImage ? (
                  <div className="space-y-3">
                    <img src={selectedImage} alt="Preview" className="mx-auto max-h-48 rounded-lg" />
                    <button
                      type="button"
                      onClick={() => setSelectedImage(null)}
                      className="text-sm text-[#E74C3C] hover:underline"
                    >
                      Remove image
                    </button>
                  </div>
                ) : (
                  <label className="cursor-pointer">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                    <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                    <p className="text-foreground font-medium">Click to upload image</p>
                    <p className="text-xs text-muted-foreground mt-1">PNG, JPG up to 10MB</p>
                  </label>
                )}
              </div>
            </div>

            {/* Description */}
            <div>
              <label className="block text-foreground mb-2">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe the waste management issue..."
                rows={6}
                className="w-full px-4 py-3 bg-input-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2ECC71] resize-none"
                required
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full bg-[#2ECC71] text-white py-3 rounded-lg font-medium hover:bg-[#27ae60] transition-colors shadow-md flex items-center justify-center gap-2"
            >
              <Send className="h-4 w-4" />
              Submit Complaint
            </button>
          </form>

          {/* Info Box */}
          <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">What happens next?</h4>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
              <li>Your complaint will be reviewed by our team</li>
              <li>We'll assign it to the nearest service unit</li>
              <li>You'll receive updates via notification</li>
              <li>Average resolution time: 24-48 hours</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Right Side - Interactive Map */}
      <div className="w-1/2 relative">
        <div
          className="absolute inset-0 bg-gradient-to-br from-blue-50 to-green-50 cursor-crosshair"
          onClick={handleMapClick}
        >
          {/* Grid overlay */}
          <div
            className="absolute inset-0"
            style={{
              backgroundImage:
                "linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)",
              backgroundSize: "50px 50px",
            }}
          />

          {/* Instructions */}
          <div className="absolute top-6 left-1/2 transform -translate-x-1/2 bg-white px-4 py-2 rounded-lg shadow-md border border-border">
            <p className="text-sm text-foreground flex items-center gap-2">
              <MapPin className="h-4 w-4 text-[#2ECC71]" />
              Click anywhere on the map to mark complaint location
            </p>
          </div>

          {/* Marker */}
          {markerPosition && (
            <div
              className="absolute transform -translate-x-1/2 -translate-y-full"
              style={{ left: `${markerPosition.x}%`, top: `${markerPosition.y}%` }}
            >
              <div className="relative">
                <MapPin className="h-10 w-10 text-[#E74C3C] drop-shadow-lg animate-bounce" fill="#E74C3C" />
                <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 bg-white px-3 py-2 rounded-lg shadow-md whitespace-nowrap text-sm font-medium">
                  Complaint Location
                </div>
              </div>
            </div>
          )}

          {/* Map Info */}
          <div className="absolute bottom-6 right-6 bg-white rounded-lg p-4 shadow-lg border border-border">
            <h4 className="font-semibold text-foreground mb-2 text-sm">Map Guide</h4>
            <div className="space-y-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-[#E74C3C] rounded-full" />
                <span>Complaint marker</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="w-3 h-3 text-gray-400" />
                <span>Click to place marker</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
