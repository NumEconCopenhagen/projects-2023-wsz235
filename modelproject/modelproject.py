import numpy as np
from scipy import optimize
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, FloatSlider, IntSlider, Layout

# A class representing the Cournot model
class Cournotmodel:

    # Initialize the Cournot model with parameter values a, b, and c
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    # Compute the market price based on quantities q1 and q2
    def p(self, q1, q2):
        return self.a - self.b * (q1 + q2)

    # Compute the cost of production based on quantity q
    def cost(self, q):
        return q * self.c

    # Compute the profit of firm 1 based on quantities q1 and q2
    def firm_profit_1(self, q1, q2):
        return self.p(q1, q2) * q1 - self.cost(q1)

    # Compute the profit of firm 2 based on quantities q1 and q2
    def firm_profit_2(self, q1, q2):
        return self.p(q1, q2) * q2 - self.cost(q2)

    # Compute the best response quantity of firm 1 given q2
    def BR_1(self, q2):
        Q1 = optimize.minimize(lambda q0: -self.firm_profit_1(q0, q2), [0]).x[0]
        return max(Q1, 0)

    # Compute the best response quantity of firm 2 given q1
    def BR_2(self, q1):
        Q2 = optimize.minimize(lambda q0: -self.firm_profit_2(q1, q0), [0]).x[0]
        return max(Q2, 0)

    # Define the conditions for the equilibrium
    def conditions(self, q):
        u = q[0] - self.BR_1(q[1])
        y = q[1] - self.BR_2(q[0])
        return [u, y]
    
    # Find the equilibrium quantities, price, and profit
    def find_equilibrium(self, initial_guess):
        solver = optimize.fsolve(self.conditions, initial_guess)
        q_star = solver[0]
        p_star = self.p(q_star, q_star)
        pi_star = self.firm_profit_1(q_star, q_star)

        return q_star, p_star, pi_star

    # Calculate the production levels for firm 1 and firm 2 based on firm 1's production level
    def calculate_production_levels(self, production_level_f1):
        production_level_f2 = np.vectorize(self.BR_2)(production_level_f1)
        production_level_f1_2 = np.vectorize(self.BR_1)(production_level_f2)
        return production_level_f1_2, production_level_f2

    # Plot the scatter plot illustrating the Cournot Equilibrium
    def plot_scatter(self, production_level_f1, production_level_f2):
        fig = px.scatter()
        fig.add_scatter(
            x=production_level_f2,
            y=production_level_f1,
            mode="lines",
            name="Firm 1",
            line=dict(color="red"),
        )
        fig.add_scatter(
            x=production_level_f1,
            y=production_level_f2,
            mode="lines",
            name="Firm 2",
            line=dict(color="blue"),
        )

        fig.update_layout(
            title="Cournot Equilibrium, BR function for firm 1 and firm 2"
        )
        fig.update_xaxes(
            title="Output of firm 1",
            range=[0, 10]  # Set the range for the x-axis
        )
        fig.update_yaxes(
            title="Output of firm 2",
            range=[0, 10]  # Set the range for the y-axis
        )

        fig.show()

    # Plot the profit functions for firm 1 and firm 2
    def plot_profit_functions(self):
        q1_values = np.linspace(0, 5, 100)
        q2_values = np.linspace(0, 5, 100)
        q1_grid, q2_grid = np.meshgrid(q1_values, q2_values)

        profit_1_grid = self.firm_profit_1(q1_grid, q2_grid)
        profit_2_grid = self.firm_profit_2(q1_grid, q2_grid)

        q_star, _, pi_star = self.find_equilibrium([1, 1])

        fig = go.Figure(data=[
            go.Surface(
                x=q1_grid,
                y=q2_grid,
                z=profit_2_grid,
                colorscale='plasma',
                opacity=0.8,
                hovertemplate='q1: %{x}<br>q2: %{y}<br>Profit: %{z}<extra></extra>',
            ),
        ])

        fig.add_trace(
            go.Scatter3d(
                x=[q_star],
                y=[q_star],
                z=[pi_star],
                mode='markers',
                marker=dict(color='red', size=5),
                name='Nash Equilibrium',
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='q1',
                yaxis_title='q2',
                zaxis_title='Profit',
            ),
            title='Profit Functions for Firm 2',
        )

        fig.show()

# A class representing the firms in the Cournot model
class Firms:

    # Initialize the firms with parameter values a, b, and c
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.profits = []  
        self.prices = [] 

    # Plot the convergence of profits and prices as the number of firms increases
    def plot_firm_convergence(self):
        for n in range(2, 101):
            q = (self.a - self.c) / (n + 1)  
            Q = n * q 
            P = self.a - Q  
            profit = (P - self.c) * q  

            self.profits.append(profit)  
            self.prices.append(P)  

        fig = go.Figure()

        for n in range(2, 101):
            fig.add_trace(go.Scatter(x=list(range(1, n+1)), y=self.profits[:n],
                                mode='lines',
                                name='Profit per Firm',
                                visible=False,
                                hovertemplate='%{y:.2f}',
                                line=dict(color='red')))
            fig.add_trace(go.Scatter(x=list(range(1, n+1)), y=self.prices[:n],
                                mode='lines',
                                name='Price',
                                visible=False,
                                hovertemplate='%{y:.2f}',
                                line=dict(color='blue')))

        fig.data[0].visible = True
        fig.data[1].visible = True

        steps = []
        # Create a step dictionary for each iteration
        for i in range(0, 200, 2):
            step = dict(
                method="update",
                args=[{"visible": [False] * 200}, # Initialize visibility to False for all traces
                      {"title": f"Convergence of Profits and Price for {i//2 + 1} Firms"}], # Set the title for the step
                label=str(i // 2 + 1) # Set the label for the slider step
            )
            step["args"][0]["visible"][i] = True    # Toggle the visibility of the i'th trace to True
            step["args"][0]["visible"][i+1] = True  # Toggle the visibility of the (i+1)'th trace to True
            steps.append(step)                      # Append the step to the list of steps

        sliders = [dict(
            active=1,
            currentvalue={"prefix": "Number of Firms: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            title="Convergence of Profits and Price in a Cournot Model",
            xaxis_title="Number of Firms",
            yaxis_title="Value",
        )

        fig.show()
