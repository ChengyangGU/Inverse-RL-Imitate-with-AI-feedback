import numpy as np


def CreateSystemInfoPrompt():

    message_start = f'''You need to assist optimal planning of a Cliff Walking problem.\n 
    Here is some information about state and action variables involved in this system:\n'''
    sys = f'''This  Cliff Walking problem involves crossing a 4*12 gridworld.
    The player in this game has 4 feasible actions:
    0: Move Up 
    1: Move Right 
    2: Move Down 
    3: Move Left 
    The state in this game records current location [row, col] of the player with a number row*12+col. 
    The game starts with the player at location [3, 0] (i.e. state 36). The goal is located at [3, 11] (i.e. state 47).
    A cliff runs along [3, 1..10](i.e. state 37-46). If the player moves to a cliff location it returns to the start location.
    The maximum move steps of the player is 100.\n
    '''
    obj = f'''
    In this game, our objective is to find the path with fewest move steps from start to goal while avoiding falling in the cliff region.\n
    '''
    message_end = f'''Please remember the above information.'''
    message = message_start + sys + obj + message_end

    return message


def CreatePreferenceEvalPrompt(input):
    message_start = f'''Now, you need to based on above system and objective information to evaluate 2 rollouts (i.e. path plans) for the player.\
    The format is:\n
    Rollout 1: \n
    action_rollout: [action(0),...,action(99)]\n
    state_rollout: [state(0),...,state(99)]\n
    Rollout 2: \n
    action_rollout: [action(0),...,action(99)]\n
    state_rollout: [state(0),...,state(99)]\n\n'''
    message_mid = f'''Here are the 2 rollouts you need to evaluate:\n\n'''

    message_end =f'''Please choose your preferred rollout. You should not give any explanations. Just Answer with 1 or 2.'''

    message = message_start + message_mid + input + message_end
    return message

def Load2Rollouts(action_rollout, state_rollout, k):
    message_start1=f'''Rollout 1: \n'''
    message_start2=f'''Rollout 2: \n'''
    str_action1 = 'action_rollout:['
    str_action2 = 'action_rollout:['
    str_state1 = 'state_rollout:['
    str_state2 = 'state_rollout:['


    for i in range(len(action_rollout[82])):
        str_action1 = str_action1 + str(action_rollout[2*k][i])
        str_action1 = str_action1 + ','



        str_state1 = str_state1 + str(state_rollout[2*k][i])
        str_state1 = str_state1 + ','

    for i in range(len(action_rollout[83])):
        str_action2 = str_action2 + str(action_rollout[2 * k + 1][i])
        str_action2 = str_action2 + ','

        str_state2 = str_state2 + str(state_rollout[2*k+1][i])
        str_state2 = str_state2 + ','



    message = message_start1 + str_action1 + ']\n'+ str_state1 + ']\n' + '\n'+ message_start2 +  str_action2 +']\n'+ str_state2 +']\n'
    return message

if __name__ == '__main__':
    k = 41
    action_rollout = np.load('random_action.npy', allow_pickle=True)
    state_rollout = np.load('random_state.npy', allow_pickle=True)

    #print(len(action_rollout))
    pair_input = Load2Rollouts(action_rollout, state_rollout, k)
    print(CreatePreferenceEvalPrompt(pair_input))

