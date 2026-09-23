import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query";

import { handleUnauthorized } from "../features/admin/session";

export function createQueryClient() {
  const client: QueryClient = new QueryClient({
    queryCache: new QueryCache({ onError: (error) => handleUnauthorized(client, error) }),
    mutationCache: new MutationCache({ onError: (error) => handleUnauthorized(client, error) }),
    defaultOptions: {
      queries: { refetchOnWindowFocus: false, retry: false },
    },
  });
  return client;
}
