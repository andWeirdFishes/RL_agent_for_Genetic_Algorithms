import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy


class VRPSolution:
    def __init__(self,
                 routes: List[List[int]],
                 instance_data: Dict,
                 cost: Optional[float] = None,
                 is_feasible: Optional[bool] = None):
        self.routes = routes
        self.instance_data = instance_data
        self._cost = cost
        self._is_feasible = is_feasible
        self.distance_matrix = instance_data['distance_matrix']
        self.demands = instance_data['demands']
        self.capacity = instance_data['capacity']
        self.num_customers = instance_data['dimension'] - 1
        self.num_vehicles = instance_data.get('num_vehicles', len(routes))

    @property
    def cost(self) -> float:
        if self._cost is None:
            self._cost = self.calculate_cost()
        return self._cost

    @property
    def is_feasible(self) -> bool:
        if self._is_feasible is None:
            self._is_feasible = self.check_feasibility()
        return self._is_feasible

    def calculate_cost(self) -> float:
        return sum(self.calculate_route_cost(route) for route in self.routes)

    def calculate_route_cost(self, route: List[int]) -> float:
        if not route:
            return 0.0
        route_cost = self.distance_matrix[0, route[0]]
        for i in range(len(route) - 1):
            route_cost += self.distance_matrix[route[i], route[i + 1]]
        route_cost += self.distance_matrix[route[-1], 0]
        return route_cost

    def calculate_route_demand(self, route: List[int]) -> int:
        return sum(self.demands[customer] for customer in route)

    def check_feasibility(self) -> bool:
        visited = set()
        for route in self.routes:
            if len(route) == 0:
                return False
            for customer in route:
                if customer in visited or customer == 0:
                    return False
                visited.add(customer)
            route_demand = self.calculate_route_demand(route)
            if route_demand > self.capacity:
                return False
        for customer in range(1, self.num_customers + 1):
            if customer not in visited:
                return False
        return True

    def get_route_demands(self) -> List[int]:
        return [self.calculate_route_demand(route) for route in self.routes]

    def get_route_costs(self) -> List[float]:
        return [self.calculate_route_cost(route) for route in self.routes]

    def clone(self) -> 'VRPSolution':
        return VRPSolution(
            routes=deepcopy(self.routes),
            instance_data=self.instance_data,
            cost=self._cost,
            is_feasible=self._is_feasible
        )

    def invalidate_cache(self) -> None:
        self._cost = None
        self._is_feasible = None

    def to_giant_tour(self) -> List[int]:
        giant_tour = []
        for route in self.routes:
            giant_tour.extend(route)
        return giant_tour

    @classmethod
    def from_giant_tour(cls, giant_tour: List[int], instance_data: Dict,
                        num_routes: Optional[int] = None) -> 'VRPSolution':
        if num_routes is not None:
            routes = [list(x) for x in np.array_split(giant_tour, num_routes) if len(x) > 0]
            return cls(routes=routes, instance_data=instance_data)

        routes = []
        current_route = []
        current_demand = 0
        capacity = instance_data['capacity']
        demands = instance_data['demands']
        for customer in giant_tour:
            customer_demand = demands[customer]
            if current_demand + customer_demand <= capacity:
                current_route.append(customer)
                current_demand += customer_demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_demand = customer_demand
        if current_route:
            routes.append(current_route)
        return cls(routes=routes, instance_data=instance_data)

    def __repr__(self) -> str:
        return f"VRPSolution(routes={len(self.routes)}, cost={self.cost:.2f}, feasible={self.is_feasible})"

    def __str__(self) -> str:
        result = [f"VRP Solution (Cost: {self.cost:.2f}, Feasible: {self.is_feasible})"]
        for i, route in enumerate(self.routes, 1):
            demand = self.calculate_route_demand(route)
            route_cost = self.calculate_route_cost(route)
            result.append(f"  Route {i}: {route} (Demand: {demand}/{self.capacity}, Cost: {route_cost:.2f})")
        return '\n'.join(result)