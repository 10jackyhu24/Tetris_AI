def action_decoder(code) -> list: # code: (is_use_hold, rotate, move)
    actions = []

    if code[0]:
        actions.append('C_down')

    for rotation in range(abs(code[1])):
        if code[1] > 0:
            actions.append('X_down')
        elif code[1] < 0:
            actions.append('Z_down')

    for movement in range(abs(code[2])):
        if code[2] > 0:
            actions.append('Right_down')
        elif code[2] < 0:
            actions.append('Left_down')

    actions.append('Down_down')
    actions.append('space_down')

    return actions