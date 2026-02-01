from classifier import Classifier


class Compute:
    def __init__(self, setup: Classifier):
        self.setup = setup

    def print_details(self):
        team1, team2 = self.setup.teams
        print(f"Team 1 is colored {team1.team_color}\nTeam 2 is colored {team2.team_color}")
        print(f"Team 1 has {len(team1.players)} players\nTeam 2 has {len(team2.players)} players")
