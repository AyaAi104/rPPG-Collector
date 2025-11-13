# gui.py
import tkinter as tk
from tkinter import scrolledtext, messagebox
import queue
import threading


class AppGUI(tk.Tk):
    """
    A simple GUI application to interact with the PulseSensorCollector.
    It provides a text area for output, an entry box for custom commands,
    and dedicated buttons for common actions.
    """

    def __init__(self, command_queue, collector_instance):
        super().__init__()
        self.title("Pulse Sensor Control Panel")
        self.geometry("800x600")

        self.command_queue = command_queue
        self.collector = collector_instance

        # --- UI Element Creation ---

        # Create a main frame to hold all control widgets at the bottom.
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Create the text entry widget for custom commands.
        self.command_entry = tk.Entry(control_frame, font=("Helvetica", 12))
        # Bind the <Return> (Enter) key to the send_command method.
        self.command_entry.bind("<Return>", self.send_custom_command)
        # Pack the entry widget to the left, allowing it to expand and fill available space.
        self.command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        # --- NEW BUTTONS ---
        # Create dedicated buttons for 'Start', 'Pause', and 'Collect'.
        # Note: We pack them from right to left to get the desired order.

        # Create the 'Send' button for the text entry.
        self.send_button = tk.Button(control_frame, text="Send Custom", command=self.send_custom_command,
                                     font=("Helvetica", 10))
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Create the 'Collect' button.
        self.collect_button = tk.Button(control_frame, text="Collect",
                                        command=lambda: self.send_predefined_command('collect'), font=("Helvetica", 10),
                                        bg="#D9EAD3")
        self.collect_button.pack(side=tk.RIGHT, padx=5)

        # Create the 'Pause' button.
        self.pause_button = tk.Button(control_frame, text="Pause",
                                      command=lambda: self.send_predefined_command('pause'), font=("Helvetica", 10),
                                      bg="#FFF2CC")
        self.pause_button.pack(side=tk.RIGHT, padx=5)

        # Create the 'Start' button.
        self.start_button = tk.Button(control_frame, text="Start",
                                      command=lambda: self.send_predefined_command('start'), font=("Helvetica", 10),
                                      bg="#CFE2F3")
        self.start_button.pack(side=tk.RIGHT, padx=5)

        # Create a scrolled text area to display output from the collector.
        self.output_text = scrolledtext.Text(self, wrap=tk.WORD, state='disabled', font=("Consolas", 10))
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Set the protocol for the window's close button ('X').
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def send_predefined_command(self, command):
        """
        Sends a fixed command string to the collector's command queue.
        Used by the dedicated buttons (Start, Pause, Collect).
        """
        if command:
            self.command_queue.put(command)

    def send_custom_command(self, event=None):
        """
        Gets the command from the entry box, sends it to the queue,
        and clears the entry box.
        """
        command = self.command_entry.get()
        if command:
            self.command_queue.put(command)
            self.command_entry.delete(0, tk.END)

    def write(self, text):
        """
        Allows this class to act like a file-like object (e.g., for stdout redirection).
        Writes the given text to the output text area.
        """
        if self.output_text:
            self.output_text.configure(state='normal')
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)  # Auto-scroll to the bottom.
            self.output_text.configure(state='disabled')

    def flush(self):
        """
        Placeholder method for stdout redirection compatibility.
        Tkinter's Text widget does not require flushing.
        """
        pass

    def on_closing(self):
        """
        Handles the window closing event. Prompts the user for confirmation
        and sends the 'quit' command to gracefully shut down the backend.
        """
        if messagebox.askokcancel("Quit", "Do you want to exit the application?"):
            # Send the 'quit' command to allow the backend to clean up.
            self.command_queue.put('quit')
            # Wait a moment for the backend to process the shutdown before destroying the window.
            self.after(500, self.destroy)


def start_gui(collector_instance):
    """
    The main entry point to create and run the GUI.
    """
    # The GUI runs in the main thread. It uses a queue to communicate
    # with the collector running in a separate background thread.
    command_queue = collector_instance.command_queue

    app = AppGUI(command_queue, collector_instance)

    # Redirect stdout and stderr so that all 'print' statements
    # appear in the GUI's text area instead of the console.
    import sys
    sys.stdout = app
    sys.stderr = app

    # Run the collector's main loop in a separate daemon thread
    # to prevent it from blocking the GUI.
    collector_thread = threading.Thread(target=collector_instance.run, daemon=True)
    collector_thread.start()

    # Start the Tkinter event loop. This is a blocking call.
    app.mainloop()