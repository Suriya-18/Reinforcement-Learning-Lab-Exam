    def chooseAction(self, state, policy, exploreRate):
      if exploreRate > np.random.rand():
        self.explored += 1
        return np.random.choice(self.actions[state])
      self.exploited += 1
      return policy[state]
     ##Read the algorithm carefully and write the code
    ''' Step 1: Generate a random number between 0 and 1 using np.random.rand().
    Step 2: Compare the random number with exploreRate.
    Step 3: If the random number is less than exploreRate (Exploration):
    Step 3.1: Increment exploration counter (self.explored += 1).
    Step 3.2: Select and return a random action from self.actions[state] using np.random.choice().
    Step 4: If the random number is greater than or equal to exploreRate (Exploitation):
    Step 4.1: Increment exploitation counter (self.exploited += 1).
    Step 4.2: Return the action from the current policy (policy[state]).'''


    def greedyChoose(self, state, values):
      actions = self.actions[state]
      stateValues = []
      for act in actions:
        row,column=self.getNewState(state,act)
        if (row, column) in values:
          stateValues.append(values[(row, column)])
      return actions[np.argmax(stateValues)]
       ##Read the algorithm carefully and write the code
    ''' Step 1: Retrieve available actions for the given state.
        Step 2: Initialize an empty list stateValues = [] to store values of possible next states.
        Step 3: For each possible action:
        Step 3.1: Compute next state using getNewState(state, action).
        Step 3.2: If the next state exists in values, store its value in stateValues.
        Step 4: Return the action that leads to the highest state value using np.argmax(stateValues).'''


    def move(self, state, policy, exploreRate):
      action = self.chooseAction(state, policy, exploreRate)
      row,column=self.getNewState(state,action)
      if (row, column) in self.rewards:
        return (row, column),self.rewards[(row, column)]
      return (row, column), 0
        ##Read the algorithm carefully and write the code
    '''
        Step 1: Select an action using chooseAction(state, policy, exploreRate).
        Step 2: Compute the new state using getNewState(state, action).
        Step 3: Check if the new state has a defined reward.
        Step 3.1: If yes, return the new state and its reward.
        Step 3.2: If no, return the new state with reward 0.'''
