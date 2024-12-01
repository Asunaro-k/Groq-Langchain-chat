# todo-with-timer
## Overview
The TODO Management System is a web application built using Streamlit and SQLite.
It allows users to manage tasks with estimated times and track elapsed times using a timer.
Users can add, view, update, and delete tasks through an interactive web interface.
The system features automatic timer updates and ensures accurate time tracking for running tasks.

## Project Structure
```
todo-with-timer/
    ├── docker-compose.yml
    ├── Dockerfile
    └── app/
        ├── app.py
        └── data/
            └── tasks.db
```

## Running the Application

To start the TODO Management System, use the following command:

```bash
docker-compose up -d
```

## Features
1. **Database Initialization**:
    - Initializes an SQLite database to store tasks if it doesn't already exist.
    - Creates a table `tasks` with columns for task ID, name, estimated time, elapsed time, running status, and start time.

2. **Task Management**:
    - **Add Task**: Allows users to add new tasks with a name and estimated time.
    - **Load Tasks**: Retrieves all tasks from the database.
    - **Update Task**: Updates task details such as elapsed time, running status, and start time.
    - **Delete Task**: Deletes a task from the database.

3. **Timer Functionality**:
    - **Update Timer**: Updates the elapsed time for running tasks.
    - **Start/Stop Timer**: Allows users to start or stop the timer for each task.
    - **Reset Timer**: Resets the elapsed time for a task.

4. **User Interface**:
    - Built using Streamlit to provide an interactive web interface.
    - Displays tasks with their details and provides buttons to control the timer and delete tasks.
    - Automatically updates the timer for running tasks and refreshes the interface.

5. **Periodic Timer Update**:
    - Checks if any task is running and updates the timer every second, ensuring accurate time tracking.

## Future Features

1. **User Authentication**:
    - Implement user login and registration to allow multiple users to manage their own tasks.

2. **Task Categories**:
    - Allow users to categorize tasks and filter them based on categories.

3. **Task Prioritization**:
    - Enable users to set priority levels for tasks and sort them accordingly.

4. **Notifications**:
    - Add notifications to remind users of upcoming deadlines or tasks that have been running for too long.

5. **Reporting and Analytics**:
    - Provide reports and analytics on task completion times and user productivity.

6. **Mobile Compatibility**:
    - Optimize the web interface for mobile devices to allow task management on the go.

7. **Integration with Calendar Apps**:
    - Integrate with popular calendar applications to sync tasks and deadlines.

8. **Collaboration Features**:
    - Allow users to share tasks and collaborate with others in real-time.

9. **Customizable Themes**:
    - Provide options for users to customize the appearance of the web interface.

10. **Voice Commands**:
    - Implement voice command functionality to add, update, and manage tasks hands-free.