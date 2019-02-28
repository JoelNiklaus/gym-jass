# gym-jass
This is a gym environment for the Schieber variant of the Swiss Card Game Jass.

# Brief description
The jass server (pyschieber) is requesting a card from the player which invokes the choose_card method in the jass client. But in the gym environment, the rl algorithm wants to initiate control by invoking the step function. So when the rl player wants to make a step, he provides an action. This action can only be processed when the jass server is waiting for input, aka when he requested a card. Then the card is sent to the jass server. The jass server computes the stich with the other players and then sends the current state back to the client (aka the observation). This observation is returned from the step function of the gym environment. 
