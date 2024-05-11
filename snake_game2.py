import numpy as np
import cv2
import random
import time


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 26) * 10, random.randrange(1, 26) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 260 or snake_head[0] < 0 or snake_head[1] >= 260 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


def get_relative_positions(snake_head, snake_position, apple_position):
    # Apple x and y positions relative to head [-1: left, 0: level, 1: right]
    if apple_position[0] < snake_head[0]:  # apple to the left of head
        apple_x = -1
    elif apple_position[0] > snake_head[0]:  # apple to the right of head
        apple_x = 1
    else:
        apple_x = 0  # apple on same row as head

    if apple_position[1] < snake_head[1]:  # apple is above head
        apple_y = -1
    elif apple_position[1] > snake_head[1]:  # apple is below head
        apple_y = 1
    else:
        apple_y = 0  # apple is on same column as head

    # tail and midsection relative to head [-1: left, 0: level, 1: right]
    # tail
    tail = snake_position[-1]
    middle = snake_position[len(snake_position) // 2]

    if tail[0] < snake_head[0]:  # tail to the left of head
        tail_x = -1
    elif tail[0] > snake_head[0]:  # tail to the right of head
        tail_x = 1
    else:
        tail_x = 0  # tail on same row as head

    if tail[1] < snake_head[1]:  # tail is above head
        tail_y = -1
    elif tail[1] > snake_head[1]:  # tail is below head
        tail_y = 1
    else:
        tail_y = 0  # tail is on same column as head

    # middle
    if middle[0] < snake_head[0]:  # middle to the left of head
        middle_x = -1
    elif middle[0] > snake_head[0]:  # middle to the right of head
        middle_x = 1
    else:
        middle_x = 0  # middle on same row as head

    if middle[1] < snake_head[1]:  # middle is above head
        middle_y = -1
    elif middle[1] > snake_head[1]:  # middle is below head
        middle_y = 1
    else:
        middle_y = 0  # middle is on same column as head

    return [apple_x, apple_y, middle_x, middle_y, tail_x, tail_y]


def scan_danger(snake_position, snake_head):
    danger_dir = {
        "up": np.array([0, -10]),
        "down": np.array([0, 10]),
        "left": np.array([-10, 0]),
        "right": np.array([10, 0]),
    }

    danger = {"up": 0, "down": 0, "left": 0, "right": 0}

    for direction in danger_dir:
        for i in range(-1, 5):
            level = i + 2
            head = np.array(snake_head)
            scanning = list((level * danger_dir[direction]) + snake_head)
            if (scanning in list(snake_position) or (scanning[0] == -10) or (scanning[1] == -10)
                or (scanning[0] == 260) or (scanning[1] == 260)):
                danger[direction] = i

                print("")
                print(direction)
                print(scanning)
                print(snake_position)
                break
            else:
                danger[direction] = 4

    return list(danger.values())

img = np.zeros((260, 260, 3), dtype='uint8')
# Initial Snake and Apple position
snake_position = [[130, 130], [120, 130], [110, 130]]
apple_position = [random.randrange(1, 26) * 10, random.randrange(1, 26) * 10]
score = 0
prev_button_direction = 1
button_direction = 1
snake_head = [130, 130]
while True:
    # Update the game state based on the last input
    if button_direction == 1:
        snake_head[0] += 10
    elif button_direction == 0:
        snake_head[0] -= 10
    elif button_direction == 2:
        snake_head[1] += 10
    elif button_direction == 3:
        snake_head[1] -= 10

    # Check if snake eats the apple
    if snake_head == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0, list(snake_head))
    else:
        snake_position.insert(0, list(snake_head))
        snake_position.pop()

    # Check for collisions
    if collision_with_boundaries(snake_head) == 1 or collision_with_self(snake_position) == 1:
        # Game over, display final score
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((260, 260, 3), dtype='uint8')
        cv2.putText(img, f'Your Score is {score}', (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('a', img)
        cv2.waitKey(0)  # Wait for any key to exit
        break

    # Draw the current state
    img = np.zeros((260, 260, 3), dtype='uint8')
    cv2.rectangle(img, (apple_position[0], apple_position[1]), (apple_position[0] + 10, apple_position[1] + 10),
                  (0, 0, 255), 3)
    for position in snake_position:
        cv2.rectangle(img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

    # Display the updated game state
    cv2.imshow('a', img)
    print(scan_danger(snake_position, snake_head))

    # Get next input
    k = cv2.waitKey(0)  # Wait indefinitely for the next key press
    if k == ord('q'):
        break  # Exit if 'q' is pressed

    # Update direction based on input
    if k == ord('a') and prev_button_direction != 1:
        button_direction = 0
    elif k == ord('d') and prev_button_direction != 0:
        button_direction = 1
    elif k == ord('w') and prev_button_direction != 2:
        button_direction = 3
    elif k == ord('s') and prev_button_direction != 3:
        button_direction = 2

    prev_button_direction = button_direction  # Save the last direction


cv2.destroyAllWindows()
