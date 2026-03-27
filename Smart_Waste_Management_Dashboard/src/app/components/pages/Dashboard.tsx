import { useEffect, useState } from "react";
import { Trash2, AlertTriangle, Truck, Bell, Loader2 } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table";
import { getDashboardStats, DashboardStats } from "../../services/adminService";
import { formatDistanceToNow } from 'date-fns';
import { useSocket } from "../../context/SocketContext";

export default function Dashboard() {
  const [data, setData] = useState<DashboardStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  const { socket } = useSocket();

  const fetchData = async () => {
      try {
        const result = await getDashboardStats();
        setData(result);
      } catch (err: any) {
        setError(err.message || "Failed to load dashboard data");
      } finally {
        setIsLoading(false);
      }
    };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (!socket) return;

    // Refresh dashboard stats on relevant events silently without triggering loading spinner
    const refreshData = async () => {
      try {
        const result = await getDashboardStats();
        setData(result);
      } catch (err) {}
    };

    socket.on('new_request', refreshData);
    socket.on('task_updated', refreshData);
    socket.on('new_complaint', refreshData);
    socket.on('complaint_updated', refreshData);
    socket.on('bin_updated', refreshData);
    socket.on('new_bin', refreshData);
    socket.on('delete_bin', refreshData);

    return () => {
       socket.off('new_request', refreshData);
       socket.off('task_updated', refreshData);
       socket.off('new_complaint', refreshData);
       socket.off('complaint_updated', refreshData);
       socket.off('bin_updated', refreshData);
       socket.off('new_bin', refreshData);
       socket.off('delete_bin', refreshData);
    }
  }, [socket]);

  if (isLoading) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-lg text-muted-foreground">Loading dashboard...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 text-center text-destructive">
        <AlertTriangle className="mx-auto h-12 w-12 mb-4" />
        <p>{error || "Failed to load data"}</p>
      </div>
    );
  }

  const stats = [
    { label: "Total Bins", value: data.stats.totalBins.toString(), icon: Trash2, color: "bg-primary text-primary-foreground", trend: "" },
    { label: "Full Bins", value: data.stats.fullBins.toString(), icon: AlertTriangle, color: "bg-destructive text-destructive-foreground", trend: "" },
    { label: "Active Trucks", value: data.stats.activeDrivers.toString(), icon: Truck, color: "bg-[var(--success-green)] text-white", trend: "" },
    { label: "Alerts", value: data.stats.alerts.toString(), icon: Bell, color: "bg-[var(--warning-yellow)] text-white", trend: "" },
  ];

  const wasteCollectionData = data.weeklyData.map(d => {
    const dateObj = new Date(d.date);
    return {
      day: dateObj.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }),
      waste: d.count
    };
  });

  return (
    <div className="p-6 space-y-6 animate-fade-in relative z-10 w-full">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} className="hover:-translate-y-1 transition-transform duration-300">
              <CardContent className="p-6 flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground font-medium mb-1">{stat.label}</p>
                  <h3 className="text-3xl font-bold text-foreground">{stat.value}</h3>
                  <p className="text-xs text-emerald-500 font-medium mt-1">{stat.trend}</p>
                </div>
                <div className={`p-3 rounded-xl shadow-sm ${stat.color}`}>
                  <Icon className="h-6 w-6" />
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Waste Collection Trend */}
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>Waste Collection Completions (Last 7 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={wasteCollectionData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorWaste" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--success-green, #10b981)" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="var(--success-green, #10b981)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
                <XAxis 
                  dataKey="day" 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fontSize: 12, fill: '#64748b' }} 
                  dy={10} 
                />
                <YAxis 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fontSize: 12, fill: '#64748b' }} 
                  dx={-10} 
                  allowDecimals={false}
                />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: "white", 
                    borderRadius: "12px", 
                    border: "none", 
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)" 
                  }}
                  itemStyle={{ color: "#10b981", fontWeight: "bold" }}
                />
                <Area 
                  type="monotone" 
                  dataKey="waste" 
                  name="Completed Collections"
                  stroke="var(--success-green, #10b981)" 
                  strokeWidth={3}
                  fillOpacity={1} 
                  fill="url(#colorWaste)" 
                  activeDot={{ r: 6, fill: "#10b981", stroke: "#fff", strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* AI Alerts Panel */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-[var(--warning-yellow)]" />
              Recent Actions Needed
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
             {data.recentComplaints.slice(0, 5).map((alert) => (
              <div
                key={alert._id}
                className={`p-3 rounded-lg border-l-4 ${
                  alert.priority === "High"
                    ? "bg-destructive/10 border-destructive"
                    : alert.priority === "Medium"
                    ? "bg-amber-500/10 border-amber-500"
                    : "bg-emerald-500/10 border-emerald-500"
                }`}
              >
                <p className="text-sm text-foreground font-medium">{alert.description}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  At {alert.location} • {formatDistanceToNow(new Date(alert.createdAt))} ago
                </p>
              </div>
            ))}
            {data.recentComplaints.length === 0 && (
              <p className="text-sm text-muted-foreground">No pending alerts.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Recent Complaints Table */}
        <Card className="xl:col-span-3 overflow-hidden">
          <CardHeader>
            <CardTitle>Recent Complaints</CardTitle>
          </CardHeader>
          <div className="px-6 pb-6 w-full overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Location</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.recentComplaints.map((complaint) => (
                  <TableRow key={complaint._id}>
                    <TableCell className="font-medium text-foreground">{complaint.location}</TableCell>
                    <TableCell className="text-muted-foreground truncate max-w-[200px]">{complaint.description}</TableCell>
                    <TableCell>
                      <span
                        className={`inline-block px-2 py-1 rounded-md text-xs font-semibold ${
                          complaint.status === "Resolved"
                            ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
                            : "bg-amber-500/10 text-amber-600 border border-amber-500/20"
                        }`}
                      >
                        {complaint.status}
                      </span>
                    </TableCell>
                    <TableCell className="text-muted-foreground text-xs">
                      {formatDistanceToNow(new Date(complaint.createdAt))} ago
                    </TableCell>
                  </TableRow>
                ))}
                {data.recentComplaints.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center py-4 text-muted-foreground">
                      No complaints found.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>
      </div>
    </div>
  );
}