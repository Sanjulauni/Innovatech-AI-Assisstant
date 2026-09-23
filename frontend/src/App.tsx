import { QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router";

import { Layout } from "./components/Layout";
import { AdminPage } from "./features/admin/AdminPage";
import { ChatPage } from "./features/chat/ChatPage";
import { createQueryClient } from "./lib/queryClient";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<ChatPage />} />
        <Route path="admin" element={<AdminPage />} />
        <Route path="*" element={<ChatPage />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  const [queryClient] = useState(createQueryClient);
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
