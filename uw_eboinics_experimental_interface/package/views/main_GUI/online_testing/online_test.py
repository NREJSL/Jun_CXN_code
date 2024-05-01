from package.entity.base.mrcp_buffer import MRCPBuffer
from package.entity.base.ssvep_buffer import SSVEPBuffer
from package.entity.base.ssvep_buffer import filter_range
from package.entity.base.scope import Scope
from package.entity.base.filter import Filter


stimulus_type_dict = {'SSVEP_Flicker': 'ssvep',
                      'SSMVEP_Checkerboard': 'ssmvep',
                      'AO_Gait': 'ao_gait'}

detection_algorithm_type_dict = {'CCA': 'cca',
                                 'C-CNN': 'c_cnn'}
class OnlineTest():
    def onClicked_pushButton_online_ssvep_task(self):
        print('Online SSVEP Task Initiated...')
        
        window_size = float(self.ui.lineEdit_InputSSVEPOnlineWindow.text())
        
        window_stride = float(self.ui.lineEdit_InputSSVEPOnlineStrideLength.text())
        
        channels_string = self.ui.lineEdit_InputOnlineSSVEPChannels.text()
        
        channels_list = channels_string.split(',')
        stimulus_type = stimulus_type_dict[self.ui.ParadigmTypeDropDownSelection.currentText()]
        detection_algorithm_type = detection_algorithm_type_dict[self.ui.DetectionAlgorithmTypeDropDownSelection.currentText()]
        stimulus_frequency_string = self.ui.lineEdit_InputSSVEPStimulusFrequencies.text()
        
        target_frequencies = list(map(float, stimulus_frequency_string.split(',')))
        filter_range=[5,17,6]
        self.bp_filter_full = Filter(low_cut=filter_range[0],hi_cut=filter_range[1], order=filter_range[2], sf=self.sr.sample_rate, n_chan=len(self.sr.ch_list)-1)
        self.EEGNET_buffer = SSVEPBuffer(self, window_stride=window_stride, window_size=window_size, buffer_size=5,
                                         l_filter=self.bp_filter_full, filter_type='bpf', 
                                         model_name=detection_algorithm_type, stimulus_type=stimulus_type, 
                                         channels_list=channels_list, target_frequencies=target_frequencies)
        
        self.EEGNET_buffer.start_timer()
        
        
    def onClicked_pushButton_test(self):
        print("hi")
        self.bp_filter_full = Filter(low_cut=1, hi_cut=17, order=2, sf=250, n_chan=31)
        self.bp_filter_low = Filter(low_cut=1, hi_cut=5, order=2, sf=250, n_chan=31)
        self.EEGNET_buffer = MRCPBuffer(self, window_stride=0.1, window_size=1.5, buffer_size=5,
                                      l_filter=self.bp_filter_full, filter_type='bpf',
                                      ica_path=r'model/sub146_ica_005_40.fiff', downsample=100,
                                      model_path=r'model/EEGNET_test_fold_0', model_type='keras', model_name='EEGNET')
        # self.SVM_buffer = MRCPBuffer(self, window_stride=0.1, window_size=2, buffer_size=10,
        #                               l_filter=self.bp_filter_low, filter_type='bpf',
        #                               ica_path=r'model/sub146_ica.fiff', downsample=100,
        #                               model_path=r'model/svm_test_fold_0.pkl', model_type='sklearn', model_name='SVM')
        # self.ETSSVM_buffer = MRCPBuffer(self, window_stride=0.1, window_size=2, buffer_size=10,
        #                               l_filter=self.bp_filter_full, filter_type='bpf',
        #                               ica_path=r'model/sub146_ica_005_40.fiff', downsample=100,
        #                               model_path=r'model/etssvm_test_fold_0.pkl', model_type='sklearn', model_name='ETSSVM')

        self.EEGNET_scope = Scope(self.EEGNET_buffer)
        # self.SVM_scope = Scope(self.SVM_buffer)
        # self.ETSSVM_scope = Scope(self.ETSSVM_buffer)

        self.EEGNET_buffer.start_timer()
        # self.SVM_buffer.start_timer()
        # self.ETSSVM_buffer.start_timer()

        self.EEGNET_scope.start_timer(20)
        # self.SVM_scope.start_timer(20)
        # self.ETSSVM_scope.start_timer(20)

        self.EEGNET_buffer.mrcp_test_win.show()
        # self.SVM_buffer.mrcp_test_win.show()
        # self.ETSSVM_buffer.mrcp_test_win.show()

    def onClicked_pushButton_scope_mrcp(self):
        self.EEGNET_scope.show_win()
        # self.SVM_scope.show_win()
        # self.ETSSVM_scope.show_win()

    def change_window_stride(self):
        self.EEGNET_buffer.update_window_stride(float(self.ui.lineEdit_window_stride_mrcp.text()))
        # self.SVM_buffer.update_window_stride(float(self.ui.lineEdit_window_stride_mrcp.text()))
        # self.ETSSVM_buffer.update_window_stride(float(self.ui.lineEdit_window_stride_mrcp.text()))