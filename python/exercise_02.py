# -*- coding: utf-8 -*-
"""
Exercise 2: A basic Brain-Computer Interface
=============================================

Description:
In this second exercise, we will learn how to use an automatic algorithm to
recognize somebody's mental states from their EEG. We will use a classifier,
i.e., an algorithm that, provided some data, learns to recognize patterns,
and can then classify similar unseen information.

"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import time
import datetime
from statistics import mode

import bci_workshop_tools as BCIw  # Our own functions for the workshop

def get_unique_filename(user_identifier, activity_index):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eeg_data_{user_identifier}_activity{activity_index}_{timestamp_str}.npy"
    return filename

def record_data(inlet, index_channel, fs, buffer_length, epoch_length, overlap_length, shift_length, training_length):
    user_identifier = input("Enter user's identifier (e.g., name or participant ID): ")
    activity_index = 0  # Record only one activity

    print(f'\nRecording data for activity {activity_index}...')
    time.sleep(1)  # Pause for better user experience

    eeg_data, timestamps = inlet.pull_chunk(
            timeout=training_length + 1, max_samples=fs * training_length)
    eeg_data = np.array(eeg_data)[:, index_channel]

    # Divide data into epochs
    eeg_epochs = BCIw.epoch(eeg_data, epoch_length * fs, overlap_length * fs)

    # Compute features
    feat_matrix = BCIw.compute_feature_matrix(eeg_epochs, fs)

    # Save data to file
    filename = get_unique_filename(user_identifier, activity_index)
    np.save(filename, feat_matrix)  # Save feature matrix to a file
    print(f'Data for activity {activity_index} saved in {filename}')

    return eeg_epochs

def collect_and_display_decisions(inlet, index_channel, fs, buffer_length, epoch_length, overlap_length, shift_length, decision_duration):
    # Initialize the buffers for storing raw EEG and decisions
    eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
    filter_state = None  # for use with the notch filter
    decision_buffer = np.zeros((int(fs * decision_duration), 1))

    # The try/except structure allows to quit the while loop by aborting the script with <Ctrl-C>
    print(f'Collecting decisions for {decision_duration} seconds. Press Ctrl-C to stop.')

    try:
        start_time = time.time()
        while time.time() - start_time < decision_duration:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(shift_length * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, index_channel]

            # Update EEG buffer
            eeg_buffer, filter_state = BCIw.update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

            """ 3.2 COMPUTE FEATURES AND CLASSIFY """
            # Get newest samples from the buffer
            data_epoch = BCIw.get_last_data(eeg_buffer,
                                            epoch_length * fs)

            # Compute features
            feat_vector = BCIw.compute_feature_vector(data_epoch, fs)
            y_hat = BCIw.test_classifier(classifier,
                                         feat_vector.reshape(1, -1), mu_ft,
                                         std_ft)

            decision_buffer, _ = BCIw.update_buffer(decision_buffer,
                                                    np.reshape(y_hat, (-1, 1)))

            # Display elapsed time
            elapsed_time = time.time() - start_time
            print(f'Time elapsed: {elapsed_time:.2f} seconds', end='\r')
            print(y_hat)

        print('\nDecision collection complete!')

        """ 3.3 PROCESS DECISIONS """
        # Convert decision buffer to a 1D array
        decisions = decision_buffer.flatten()
        # print(decisions)
        actual = len(decisions) // 2
        midpoint = decisions[actual:]
        print(midpoint)

        # Calculate the most frequently occurring value
        most_frequent_value = mode(midpoint)

        # Display the result
        print(f'\nMost frequently occurring decision: {most_frequent_value}')

    except KeyboardInterrupt:
        print('\nDecision collection interrupted.')



if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL stream
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info, description, sampling frequency, number of channels
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()

    # Get names of all channels
    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, n_channels):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    """ 2. SET EXPERIMENTAL PARAMETERS """

    # Length of the EEG data buffer (in seconds)
    # This buffer will hold last n seconds of data and be used for calculations
    buffer_length = 15

    # Length of the epochs used to compute the FFT (in seconds)
    epoch_length = 1

    # Amount of overlap between two consecutive epochs (in seconds)
    overlap_length = 0.8

    # Amount to 'shift' the start of each next consecutive epoch
    shift_length = epoch_length - overlap_length

    # Index of the channel (electrode) to be used
    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
    index_channel = [0, 1, 2, 3]
    # Name of our channel for plotting purposes
    ch_names = [ch_names[i] for i in index_channel]
    n_channels = len(index_channel)

    # Get names of features
    # ex. ['delta - CH1', 'pwr-theta - CH1', 'pwr-alpha - CH1',...]
    feature_names = BCIw.get_feature_names(ch_names)

    # Number of seconds to collect training data for (one class)
    training_length0 = 30

    """ 3. RECORD TRAINING DATA """
    eeg_epochs0 = record_data(inlet, index_channel, fs, buffer_length, epoch_length, overlap_length, shift_length, training_length0)
    print(f'Waiting for {training_length0} seconds before recording data for the second user...')
    training_length1 = 30
    eeg_epochs1 = record_data(inlet, index_channel, fs, buffer_length, epoch_length, overlap_length, shift_length, training_length1)
    # training_length2 = 20
    # eeg_epochs2 = record_data(inlet, index_channel, fs, buffer_length, epoch_length, overlap_length, shift_length, training_length2)
    """ 4. COMPUTE FEATURES AND TRAIN CLASSIFIER """
    


    feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, fs)
    feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, fs)
    # feat_matrix2 = BCIw.compute_feature_matrix(eeg_epochs2, fs)

    feat_matrix0 = np.nan_to_num(feat_matrix0)
    feat_matrix1 = np.nan_to_num(feat_matrix1)
    # feat_matrix2 = np.nan_to_num(feat_matrix2)

    [classifier, mu_ft, std_ft] = BCIw.train_classifier(
            feat_matrix0, feat_matrix1, 'SVM', target_type='binary')
    # [classifier, mu_ft, std_ft] = BCIw.train_multi_classifier(
    #         [feat_matrix0, feat_matrix1], 'SVM')
    #BCIw.beep()

    """ 5. USE THE CLASSIFIER IN REAL-TIME"""

    # decision_duration = 20  # Duration to collect decisions (in seconds)

    # while True:
    #     user_input = input("Enter 1 to authenticate (or 'exit' to quit): ")
        
    #     if user_input == "1":
    #         collect_and_display_decisions(inlet, index_channel, fs, buffer_length, epoch_length, overlap_length, shift_length, decision_duration)
    #     elif user_input.lower() == 'exit':
    #         print("Exiting the program.")
    #         break
    #     else:
    #         print("Invalid input. You can switch who to authenticate now.")
    #         time.sleep(20)


    #collect_and_display_decisions(inlet, index_channel, fs, buffer_length, epoch_length, overlap_length, shift_length, decision_duration)
    

    # Initialize the buffers for storing raw EEG and decisions
    eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
    filter_state = None  # for use with the notch filter
    decision_buffer = np.zeros((30, 1))

    plotter_decision = BCIw.DataPlotter(30, ['Decision'])

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(shift_length * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, index_channel]

            # Update EEG buffer
            eeg_buffer, filter_state = BCIw.update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

            """ 3.2 COMPUTE FEATURES AND CLASSIFY """
            # Get newest samples from the buffer
            data_epoch = BCIw.get_last_data(eeg_buffer,
                                            epoch_length * fs)

            # Compute features
            feat_vector = BCIw.compute_feature_vector(data_epoch, fs)
            y_hat = BCIw.test_classifier(classifier,
                                         feat_vector.reshape(1, -1), mu_ft,
                                         std_ft)
            print(y_hat)

            decision_buffer, _ = BCIw.update_buffer(decision_buffer,
                                                    np.reshape(y_hat, (-1, 1)))

            """ 3.3 VISUALIZE THE DECISIONS """
            plotter_decision.update_plot(decision_buffer)
            plt.pause(0.00001)

    except KeyboardInterrupt:

        print('Closed!')
