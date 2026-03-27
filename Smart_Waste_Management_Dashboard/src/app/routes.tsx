import { createBrowserRouter } from "react-router";
import Layout from "./components/Layout";
import UserLayout from "./components/layouts/UserLayout";
import Dashboard from "./components/pages/Dashboard";
import MapPage from "./components/pages/MapPage";
import AdminPage from "./components/pages/AdminPage";
import AIInsightsPage from "./components/pages/AIInsightsPage";
import Login from "./components/auth/Login";
import Signup from "./components/auth/Signup";
import RoleSelection from "./components/auth/RoleSelection";
import GarbageCollectionPage from "./components/pages/GarbageCollectionPage";
import UserComplaintsPage from "./components/pages/UserComplaintsPage";
import DriverMapPage from "./components/pages/DriverMapPage";
import ProtectedRoute from "./components/auth/ProtectedRoute";
import RoleBasedRedirect from "./components/auth/RoleBasedRedirect";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: RoleBasedRedirect,
  },
  {
    path: "/login",
    Component: Login,
  },
  {
    path: "/signup",
    Component: Signup,
  },
  {
    path: "/select-role",
    Component: RoleSelection,
  },
  {
    path: "/dashboard",
    element: (
      <ProtectedRoute allowedRoles={['admin']}>
        <Layout />
      </ProtectedRoute>
    ),
    children: [{ index: true, Component: Dashboard }],
  },
  {
    path: "/map",
    element: (
      <ProtectedRoute allowedRoles={['admin']}>
        <Layout />
      </ProtectedRoute>
    ),
    children: [{ index: true, Component: MapPage }],
  },
  {
    path: "/admin",
    element: (
      <ProtectedRoute allowedRoles={['admin']}>
        <Layout />
      </ProtectedRoute>
    ),
    children: [{ index: true, Component: AdminPage }],
  },
  {
    path: "/ai-insights",
    element: (
      <ProtectedRoute allowedRoles={['admin']}>
        <Layout />
      </ProtectedRoute>
    ),
    children: [{ index: true, Component: AIInsightsPage }],
  },
  {
    path: "/user-collection",
    element: (
      <ProtectedRoute allowedRoles={['user']}>
        <UserLayout />
      </ProtectedRoute>
    ),
    children: [{ index: true, Component: GarbageCollectionPage }],
  },
  {
    path: "/user-complaints",
    element: (
      <ProtectedRoute allowedRoles={['user']}>
        <UserLayout />
      </ProtectedRoute>
    ),
    children: [{ index: true, Component: UserComplaintsPage }],
  },
  {
    path: "/driver-map",
    element: (
      <ProtectedRoute allowedRoles={['driver']}>
        <DriverMapPage />
      </ProtectedRoute>
    ),
  },
]);