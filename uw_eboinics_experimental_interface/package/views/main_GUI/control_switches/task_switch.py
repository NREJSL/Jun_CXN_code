from threading import Thread
from PyQt5 import QtGui
from pycnbi import logger
from random import randint
from package.entity.edata.utils import Utils

class TaskSwitch():
    def onClicked_button_start_SV(self):
        """
        Event listener for task button
        """
        self.ui.statusBar.showMessage("Tasks started")
        self.SV_time = 0
        self.is_experiment_on = True
        self.window.show()




    def Update_SV_image(self):
        """ Update subject view GUI visual cues inlcuding Idle, focus, prepare, two, one, task"""
        # print('trial counter', self.trial_counter)
        # print('cue counter', self.cue_counter)
        # print('SV time', self.SV_time)

        if self.SV_time == 0:
            self.set_epoch_number()
            self.SV_window.label_total_trial.setText(str(len(self.protocol.trial_list)))
            # initialize SV window for first trial
            first_trial = self.protocol.trial_list[self.trial_counter]

            self.SV_window.label_current_trial.setText(str(self.trial_counter + 1))
            if first_trial.name:
                self.SV_window.label_task_content.setText(self.protocol.trial_list[self.trial_counter].name)
            if first_trial.image:
                self.SV_window.label_instruction_image.setPixmap(
                    QtGui.QPixmap(self.protocol.trial_list[self.trial_counter].image))
            if first_trial.video:
                # play video
                pass
            if first_trial.sound:
                Thread(target=self.play_sound, args=(self.protocol.trial_list[self.trial_counter].sound,)).start()
            self.event_obj.add_event(first_trial.event_number, self.router.get_server_clock())

        if self.SV_time == self.protocol.trial_list[self.trial_counter].cue_list[self.cue_counter].duration:
            self.SV_time = 0

            # Not the last cue
            if self.cue_counter < len(self.protocol.trial_list[self.trial_counter].cue_list) - 1:
                # move to next cue
                self.cue_counter += 1
                self.cue_change_flag = True

            # It's the last cue, move to next trial
            else:
                # Take a break
                # if self.trial_counter == self.protocol.break_trial_number -1:
                if (self.trial_counter + 1) % self.protocol.break_trial_number == 0:
                    self.is_experiment_on = False

                # Not the last trial
                if self.trial_counter < len(self.protocol.trial_list) - 1:
                    self.cue_change_flag = True


                    self.update_MRCP_plot()
                    # move to next trial
                    self.cue_counter = 0
                    self.trial_counter += 1
                    self.set_epoch_number()
                    next_trial = self.protocol.trial_list[self.trial_counter]
                    if self.is_experiment_on:
                        self.event_obj.add_event(next_trial.event_number, self.router.get_server_clock())

                    self.SV_window.label_current_trial.setText(str(self.trial_counter + 1))
                    if next_trial.name:
                        self.SV_window.label_task_content.setText(next_trial.name)
                    if next_trial.image:
                        self.SV_window.label_instruction_image.setPixmap(
                            QtGui.QPixmap(next_trial.image))
                    if next_trial.video:
                        #play video
                        pass
                    if next_trial.sound:
                        Thread(target=self.play_sound, args=(next_trial.sound,)).start()

                # It's the last trial, finish playing tasks
                else:
                    self.stop_SV()


        if self.SV_time < self.protocol.trial_list[self.trial_counter].cue_list[self.cue_counter].duration:
            if self.cue_change_flag:
                self.cue_change_flag = False
                current_cue = self.protocol.trial_list[self.trial_counter].cue_list[self.cue_counter]
                # if self.is_experiment_on:
                self.event_obj.add_event(current_cue.event_number, self.router.get_server_clock())

                if self.ui.checkBox_show_text_as_image.isChecked() and current_cue.name:
                    self.SV_window.label.setText(current_cue.name)
                if current_cue.image:
                    # show image
                    pass
                if current_cue.video:
                    # play video
                    pass
                if current_cue.sound:
                    Thread(target=self.play_sound, args=(current_cue.sound,)).start()
            self.SV_time += 1



        #
        #
        #
        #
        # if self.SV_time % self.cycle_time == 0:
        #     # print("idle")
        #     # logger.info('\nidle server clock: {}'.format(self.router.get_server_clock()))
        #     if self.task_counter < self.new_task_table.shape[0]:
        #         self.event_timestamp_list.append(
        #             [self.event_table_dictionary['Idle'], self.router.get_server_clock()])
        #
        #     event_row = [self.event_table_dictionary['Idle'], self.router.get_server_clock()]
        #     Utils.write_data_during_recording(self.event_file_path, event_row)
        #
        #     self.SV_window.label_current_trial.setText(str(self.task_counter + 1))
        #     self.SV_window.label_total_trial.setText(str(self.new_task_table.shape[0]))
        #     self.set_epoch_number()
        #
        #     # print("\nTASK COUNTER: ", self.task_counter)
        #     if self.task_counter > 0:
        #         # self.update_MRCP_plot()
        #         # update interval time
        #         if self.ui.checkBox_randomize_interval_time.isChecked():
        #             self.idle_time = randint(1, 6)
        #             self.focus_time = self.idle_time + int(self.ui.focusTimeLineEdit.text())
        #             self.prepare_time = self.focus_time + int(self.ui.prepareTimeLineEdit.text())
        #             self.two_time = self.prepare_time + int(self.ui.twoTimeLineEdit.text())
        #             self.one_time = self.two_time + int(self.ui.oneTimeLineEdit.text())
        #             self.task_time = self.one_time + int(self.ui.taskTimeLineEdit.text())
        #             self.relax_time = self.task_time + 2
        #             self.cycle_time = self.relax_time
        #
        #     self.update_SV_task()
        #     # self.task_counter += 1
        #
        #     # Idle
        #     # self.SV_window.LBimage.setPixmap(QtGui.QPixmap("%s/package/views/icon/idle.png" % os.getcwd()))
        #     self.SV_window.label.setStyleSheet("color: green;")
        #     self.SV_window.label.setText("IDLE")
        #
        # elif self.SV_time % self.cycle_time == self.idle_time:
        #     # print("focus")
        #     # logger.info('\nfocus server clock: %s' % self.router.get_server_clock())
        #     self.event_timestamp_list.append(
        #         [self.event_table_dictionary['Focus'], self.router.get_server_clock()])
        #
        #     event_row = [self.event_table_dictionary['Focus'], self.router.get_server_clock()]
        #     Utils.write_data_during_recording(self.event_file_path, event_row)
        #     self.SV_window.label.setStyleSheet("color: blue;")
        #     self.SV_window.label.setText("FOCUS")
        # elif self.SV_time % self.cycle_time == self.focus_time:
        #     # print("prepare")
        #     # logger.info('\nprepare server clock: %s' % self.router.get_server_clock())
        #     self.event_timestamp_list.append(
        #         [self.event_table_dictionary['Prepare'], self.router.get_server_clock()])
        #
        #     event_row = [self.event_table_dictionary['Prepare'], self.router.get_server_clock()]
        #     Utils.write_data_during_recording(self.event_file_path, event_row)
        #
        #     self.SV_window.label.setStyleSheet("color: black;")
        #     self.SV_window.label.setText("PREPARE")
        # elif self.SV_time % self.cycle_time == self.prepare_time:
        #     # print("two")
        #     # logger.info('\ntwo server clock: %s' % self.router.get_server_clock())
        #     self.event_timestamp_list.append(
        #         [self.event_table_dictionary['Two'], self.router.get_server_clock()])
        #     event_row = [self.event_table_dictionary['Two'], self.router.get_server_clock()]
        #     Utils.write_data_during_recording(self.event_file_path, event_row)
        #     self.SV_window.label.setText("TWO")
        # elif self.SV_time % self.cycle_time == self.two_time:
        #     # print("one")
        #     # One
        #     # logger.info('\none server clock: %s' % self.router.get_server_clock())
        #     self.event_timestamp_list.append(
        #         [self.event_table_dictionary['One'], self.router.get_server_clock()])
        #
        #     event_row = [self.event_table_dictionary['One'], self.router.get_server_clock()]
        #     Utils.write_data_during_recording(self.event_file_path, event_row)
        #
        #     self.SV_window.label.setText("ONE")
        # elif self.SV_time % self.cycle_time == self.one_time:
        #     # print("task")
        #     # Task
        #     # logger.info('\ntask server clock: %s' % self.router.get_server_clock())
        #     # self.event_timestamp_list.append(
        #     #     [self.event_table_dictionary[self.new_task_table[self.task_counter - 1][0]],
        #     #      self.router.get_server_clock()])
        #     self.event_timestamp_list.append(
        #         [self.event_table_dictionary[self.new_task_table[self.task_counter][0]],
        #          self.router.get_server_clock()])
        #
        #     event_row = [self.event_table_dictionary[self.new_task_table[self.task_counter][0]],\
        #                  self.router.get_server_clock()]
        #     Utils.write_data_during_recording(self.event_file_path, event_row)
        #
        #     # self.SV_window.LBimage.setPixmap(QtGui.QPixmap("%s/package/views/icon/task.png" % os.getcwd()))
        #     self.SV_window.label.setStyleSheet("color: red;")
        #     self.SV_window.label.setText("TASK")
        # elif self.SV_time % self.cycle_time == self.task_time:
        #     # print("relax")
        #     # relax
        #     # self.SV_window.LBimage.setPixmap(QtGui.QPixmap("%s/package/views/icon/idle.png" % os.getcwd()))
        #     self.SV_window.label.setStyleSheet("color: green;")
        #     self.SV_window.label.setText("IDLE")
        #     self.SV_time = -2
        #
        #
        #
        # elif self.SV_time % self.cycle_time == self.task_time + 1:
        #     self.update_MRCP_plot()
        #     # add task counter and check for break
        #     # if self.task_counter < self.new_task_table.shape[0]:
        #     self.task_counter += 1
        #     self.break_trial_number = int(self.ui.lineEdit_break_trial_number.text())
        #     if self.task_counter % self.break_trial_number == 0 and self.task_counter != 0:
        #         # self.window.hide()
        #         self.is_experiment_on = False
        #     if self.task_counter >= self.new_task_table.shape[0]:
        #         logger.info('SV stopped')
        #         self.stop_SV()
        #
        # self.SV_time += 1

    def update_SV_task(self):
        """ Update task instruction image, task description and task sound on subject view GUI"""
        # Update SV UI according to task list
        if self.task_counter < self.new_task_table.shape[0]:
            self.SV_window.label_task_content.setText(self.new_task_table[self.task_counter][1])
            self.SV_window.label_instruction_image.setPixmap(QtGui.QPixmap(self.new_task_table[self.task_counter][2]))
            # self.play_task_sound(self.new_task_table[self.task_counter][3])
            Thread(target=self.play_task_sound, args=(self.new_task_table[self.task_counter][3],)).start()
        else:
            self.stop_SV()

    def stop_SV(self):
        """ Stop subject view window updating when tasks finished"""
        self.is_experiment_on = False

        # self.window.hide()
        self.ui.statusBar.showMessage("Tasks finished")
        try:
            self.ui.label_content_available_temp.setText(
                "{} - {}".format(self.temp_counter_list[0], self.temp_counter_list[-1]))
            self.ui.label_content_Disp_temp.setText(
                "{} - {}".format(self.temp_counter_list[0], self.temp_counter_list[-1]))
            self.ui.label_content_current_temp.setText(" ")
        except:
            logger.info('MRCP template display went wrong, but this does not affect data saving, please be patient ...')


            