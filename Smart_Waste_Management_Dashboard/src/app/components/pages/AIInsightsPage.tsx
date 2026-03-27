import { useState, useEffect } from "react";
import { Brain, TrendingUp, AlertCircle, Lightbulb, Activity, BarChart3, Loader2 } from "lucide-react";
import { LineChart, Line, BarChart, Bar, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { getBins, Bin } from "../../services/binService";

export default function AIInsightsPage() {
  const [bins, setBins] = useState<Bin[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchBins = async () => {
      try {
        const data = await getBins();
        setBins(data);
      } catch (error) {
        console.error("Failed to fetch bins in insights:", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchBins();
  }, []);

  const predictionData = [
    { hour: "00:00", predicted: 45, actual: 42 },
    { hour: "04:00", predicted: 38, actual: 40 },
    { hour: "08:00", predicted: 67, actual: 65 },
    { hour: "12:00", predicted: 89, actual: 87 },
    { hour: "16:00", predicted: 95, actual: 92 },
    { hour: "20:00", predicted: 78, actual: null },
    { hour: "24:00", predicted: 56, actual: null },
  ];

  const wasteTrendData = [
    { month: "Jan", residential: 1240, commercial: 890, industrial: 560 },
    { month: "Feb", residential: 1380, commercial: 920, industrial: 580 },
    { month: "Mar", residential: 1450, commercial: 980, industrial: 620 },
    { month: "Apr", residential: 1320, commercial: 850, industrial: 590 },
    { month: "May", residential: 1480, commercial: 1020, industrial: 640 },
    { month: "Jun", residential: 1560, commercial: 1100, industrial: 680 },
  ];

  const heatmapData = [
    { sector: "Sec 1", value: 95 },
    { sector: "Sec 2", value: 78 },
    { sector: "Sec 3", value: 62 },
    { sector: "Sec 4", value: 88 },
    { sector: "Sec 5", value: 45 },
    { sector: "Sec 6", value: 92 },
    { sector: "Sec 7", value: 56 },
    { sector: "Sec 8", value: 73 },
  ];

  const overflowingBins = bins.filter(b => b.fillLevel > 80);
  const criticalBin = overflowingBins.length > 0
    ? overflowingBins.reduce((prev, current) => (prev.fillLevel > current.fillLevel) ? prev : current)
    : null;

  const predictions = [
    {
      title: "Next Overflow Prediction",
      value: "Imminent",
      location: criticalBin ? (criticalBin.location?.address || `Bin ${criticalBin._id.substring(0, 6)}`) : "No immediate overflow",
      confidence: criticalBin ? "94%" : "100%",
      icon: AlertCircle,
      color: criticalBin ? "bg-red-500" : "bg-emerald-500",
    },
    {
      title: "Peak Waste Time Today",
      value: "6 PM - 8 PM",
      location: "Citywide",
      confidence: "89%",
      icon: TrendingUp,
      color: "bg-amber-500",
    },
    {
      title: "Optimal Collection Time",
      value: "Early Morning",
      location: "Sector 1-3",
      confidence: "91%",
      icon: Activity,
      color: "bg-blue-500",
    },
  ];

  const baseRecommendations = [
    {
      priority: "Medium",
      title: "Optimize Route B timing",
      description: "Current route timing misaligned with peak waste hours. Shift by 3 hours for efficiency.",
      impact: "Save 2 hours daily",
      savings: "$1,800/month",
    },
    {
      priority: "Medium",
      title: "Add bins in commercial zone",
      description: "High concentration of complaints in commercial area. 3 additional bins recommended.",
      impact: "Reduce complaints by 60%",
      savings: "$1,200/month",
    },
    {
      priority: "Low",
      title: "Seasonal adjustment needed",
      description: "Summer months show 20% increase. Plan capacity adjustment starting May.",
      impact: "Prevent service delays",
      savings: "$800/month",
    },
  ];

  const recommendations = criticalBin
    ? [
        {
          priority: "High",
          title: "Dispatch truck to " + (criticalBin.location?.address || "critical area"),
          description: `AI detected a critical bin exceeding 80% fill level (${criticalBin.fillLevel}%). Recommend immediate pickup.`,
          impact: "Prevent overflow complaints",
          savings: "$2,400/month",
        },
        ...baseRecommendations
      ]
    : baseRecommendations;

  const insights = [
    { label: "AI Model Accuracy", value: "94.2%", trend: "+2.3%" },
    { label: "Predictions Made", value: "12,847", trend: "+15%" },
    { label: "Cost Savings (MTD)", value: "$18,400", trend: "+8%" },
    { label: "Efficiency Gain", value: "23%", trend: "+5%" },
  ];

  return (
    <div className="p-6 space-y-6 animate-fade-in relative z-10 w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold text-foreground flex items-center gap-3">
            <div className="bg-primary p-3 rounded-xl shadow-lg shadow-primary/20">
              <Brain className="h-7 w-7 text-primary-foreground" />
            </div>
            AI Insights & Predictions
          </h2>
          <p className="text-muted-foreground mt-2 font-medium">Advanced analytics powered by machine learning</p>
        </div>
        <div className="bg-card px-4 py-2 rounded-lg border border-border shadow-sm">
          <p className="text-xs text-muted-foreground">Last Updated</p>
          <p className="text-sm font-semibold text-foreground">2 minutes ago</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {insights.map((insight, index) => (
          <Card key={index} className="hover:-translate-y-1 transition-transform duration-300">
            <CardContent className="p-5">
              <p className="text-sm text-muted-foreground font-medium mb-1">{insight.label}</p>
              <div className="flex items-end justify-between">
                <h3 className="text-2xl font-bold text-foreground">{insight.value}</h3>
                <span className="text-xs text-emerald-500 font-medium">{insight.trend}</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Prediction Cards */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {predictions.map((prediction, index) => {
          const Icon = prediction.icon;
          return (
            <Card key={index} className="hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-3 rounded-xl shadow-md ${prediction.color}`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <span className="text-xs bg-emerald-500/10 text-emerald-600 px-2 py-1 rounded-full font-medium border border-emerald-500/20">
                    {prediction.confidence} confident
                  </span>
                </div>
                <h4 className="font-semibold text-muted-foreground mb-2">{prediction.title}</h4>
                <p className="text-2xl font-bold text-foreground mb-3">{prediction.value}</p>
                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  {prediction.location}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction Accuracy */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-indigo-500" />
              Real-time Overflow Predictions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={predictionData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="hour" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
                />
                <Line type="monotone" dataKey="predicted" stroke="#8B5CF6" strokeWidth={3} name="Predicted" dot={{ r: 4 }} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={3} name="Actual" dot={{ r: 4 }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Waste Trend Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Waste Generation Trends
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={wasteTrendData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="month" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
                />
                <Area type="monotone" dataKey="residential" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.6} />
                <Area type="monotone" dataKey="commercial" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} />
                <Area type="monotone" dataKey="industrial" stackId="1" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Heatmap */}
        <Card className="xl:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-destructive" />
              High Waste Zone Heatmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={heatmapData} layout="vertical" margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis dataKey="sector" type="category" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--foreground)" }}
                  cursor={{ fill: 'var(--muted)' }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {heatmapData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.value > 80 ? "#EF4444" : entry.value > 60 ? "#F59E0B" : "#10B981"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* AI Recommendations */}
        <Card className="xl:col-span-2 overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="h-5 w-5 text-warning-yellow" />
              AI-Generated Recommendations
            </CardTitle>
          </CardHeader>
          <div className="px-6 pb-6 space-y-4">
            {recommendations.map((rec, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg flex flex-col md:flex-row md:items-start justify-between gap-4 border-l-4 ${
                  rec.priority === "High"
                    ? "bg-destructive/10 border-destructive"
                    : rec.priority === "Medium"
                    ? "bg-amber-500/10 border-amber-500"
                    : "bg-primary/10 border-primary"
                }`}
              >
                <div className="flex-[1]">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className={`text-[10px] uppercase tracking-wider px-2 py-1 rounded-sm font-bold ${
                        rec.priority === "High"
                          ? "bg-destructive/20 text-destructive"
                          : rec.priority === "Medium"
                          ? "bg-amber-500/20 text-amber-600"
                          : "bg-primary/20 text-primary"
                      }`}
                    >
                      {rec.priority} Priority
                    </span>
                    <h4 className="font-semibold text-foreground">{rec.title}</h4>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">{rec.description}</p>
                  <div className="flex items-center gap-4 text-xs font-semibold">
                    <span className="text-emerald-500">Impact: {rec.impact}</span>
                    <span className="text-primary">Savings: {rec.savings}</span>
                  </div>
                </div>
                <button className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-lg text-sm font-medium transition-all shadow-sm w-full md:w-auto shrink-0">
                  Apply Fix
                </button>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Model Performance */}
      <Card className="border-indigo-500/30 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-indigo-500" />
            AI Model Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-background rounded-xl p-4 border border-border shadow-sm">
              <p className="text-sm text-muted-foreground font-medium mb-1">Prediction Accuracy</p>
              <div className="flex items-end gap-2 mb-3">
                <span className="text-3xl font-bold text-foreground">94.2%</span>
                <span className="text-sm text-emerald-500 font-semibold mb-1">+2.3%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                <div className="bg-indigo-500 h-full rounded-full transition-all duration-1000" style={{ width: "94.2%" }} />
              </div>
            </div>
            
            <div className="bg-background rounded-xl p-4 border border-border shadow-sm">
              <p className="text-sm text-muted-foreground font-medium mb-1">False Positive Rate</p>
              <div className="flex items-end gap-2 mb-3">
                <span className="text-3xl font-bold text-foreground">3.8%</span>
                <span className="text-sm text-emerald-500 font-semibold mb-1">-1.2%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                <div className="bg-pink-500 h-full rounded-full transition-all duration-1000" style={{ width: "3.8%" }} />
              </div>
            </div>
            
            <div className="bg-background rounded-xl p-4 border border-border shadow-sm flex flex-col justify-center">
              <p className="text-sm text-muted-foreground font-medium mb-1">Training Dataset</p>
              <div className="flex items-end gap-2 mb-1">
                <span className="text-3xl font-bold text-foreground">2.4M</span>
                <span className="text-sm text-muted-foreground font-medium mb-1 cursor-help" title="Data Points">Points</span>
              </div>
              <p className="text-xs text-muted-foreground font-medium mt-auto border-t border-border/50 pt-2">Last trained: March 20, 2026</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}