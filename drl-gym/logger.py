import numpy as np
import os
'''
作用在于：
store_resutl(): 进行数据储存；
reveal_last():  定期输出该期间的平均值；
write_last():   定期将平均值结果写入文件，防止数据丢失；
write_final():  将每次保存的数据进行写入，不止是平均值。
'''
class Logger:
    def __init__(self):
        self.result_dict = dict()
        self.scale_dict = dict()

        self.result_iter = dict()           # used in reveal_last
        self.result_last_iter = dict()      # used in reveal_last
        self.flag_title = 1
        self.final_flag_title = 1

    def store_num(self, **kwargs):
        for k,v in kwargs.items():
            if not(k in self.scale_dict.keys()):
                self.scale_dict[k] = []
                self.result_iter[k] = 0
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            self.scale_dict[k].append(v)
            self.result_iter[k] += 1

    def store_result(self, **kwargs):     # warning: kwargs must be a dict or using "="
        for k,v in kwargs.items():
            if not(k in self.result_dict.keys()):
                self.result_dict[k] = []
                self.result_iter[k] = 0
                self.result_last_iter[k] = 0    
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            self.result_dict[k].append(v)
            self.result_iter[k] += 1

    # only reveal, without write file
    def reveal_last(self, *args):
        # auto reveal last k  mean results, where k = reveal_period in main_loop, s.t. k = 1 ...
        if len(args) > 0:
            for key in args:
                if key in self.scale_dict.keys():
                    print(str(key) , ":" , self.scale_dict[key][self.result_iter[key]-1], end=" , ")
                elif key in self.result_dict.keys():
                    value_last = np.mean(self.result_dict[key][self.result_last_iter[key]:self.result_iter[key]], axis=0)
                    print(str(key) , ":" , value_last, end=" , ")
                else:
                    raise KeyError(key)
                self.result_last_iter[key] = self.result_iter[key]
            print("\n")
        # reveal all results (lat k mean results)
        else:
            for key in self.scale_dict.keys():
                print(str(key) , ":" , self.scale_dict[key][self.result_iter[key]-1], end=" , ")
            for key in self.result_dict.keys():
                value_last = np.mean(self.result_dict[key][self.result_last_iter[key]:self.result_iter[key]], axis=0)
                print(str(key) , ":" , value_last, end=" , ")
                self.result_last_iter[key] = self.result_iter[key]
            print(" ")

    # # test... dont use
    # def reveal_last_value(self, *args):           
    #     # auto reveal last k  mean results, where k = reveal_period in main_loop, s.t. k = 1 ...
    #     if len(args) > 0:
    #         # reveal
    #         for key in args:
    #             assert key in self.result_dict.keys()
    #             value_last = np.mean(self.result_dict[key][self.result_last_iter[key]:self.result_iter[key]], axis=0)
    #             print(str(key) , ":" , value_last, end=" , ")
    #             self.result_last_iter[key] = self.result_iter[key]
    #         print(" ")
    #     # reveal all results (lat k mean results)
    #     else:
    #         for key in self.result_dict.keys():
    #             value_last = np.mean(self.result_dict[key][self.result_last_iter[key]:self.result_iter[key]], axis=0)
    #             print(str(key) , ":" , value_last, end=" , ")
    #             self.result_last_iter[key] = self.result_iter[key]
    #         print(" ")

    # only write, without reveal
    def write_last(self, save_path=os.getcwd(), save_name='result.csv', write_period=1):
        self.fl=open(save_path + '/' + save_name, 'a')
        # TODO-1: a judge --> diff save_name for diff files
        # TODO-2: only write all, next adding *args

        # write title
        if self.flag_title:
            self._write_title(self.fl, self.scale_dict)
            self._write_title(self.fl, self.result_dict)
            self.fl.write("\n")
            self.flag_title = 0
        # write value
        for key in self.scale_dict.keys():
            space_num = self.scale_dict[key][0].size   # 先把第一个数写上，后面如果有，则继续写
            if space_num == 1:
                self.fl.write(str(self.scale_dict[key][self.result_iter[key]-1]) + ",")
            else:
                for j in range(space_num):
                    self.fl.write(str(self.scale_dict[key][self.result_iter[key]-1][j]) + ",")
        for key in self.result_dict.keys():
            space_num = self.result_dict[key][0].size   # 先把第一个数写上，后面如果有，则继续写
            if space_num == 1:
                self.fl.write(str(np.mean(self.result_dict[key][-write_period:], axis=0)) + ",")
            else:
                for j in range(space_num):
                    # print(np.mean(self.result_dict[key][-write_period:], axis=0))
                    # assert True
                    self.fl.write(str(np.mean(self.result_dict[key][-write_period:], axis=0)[j]) + ",")
                    # print(str(self.result_dict[key][iter_key][j]))
                    # print(str(np.mean(self.result_dict[key], axis=0)[j]))
        self.fl.write("\n")
        # self.fl.flush()
        self.fl.close()

    def _write_title(self, file, key_list):
         for key in key_list.keys():
                # print(key_list[key][0])
                # print(key_list[key])
                # print(key)
                space_num = key_list[key][0].size
                # print(space_num)
                file.write(str(key) + ",")
                for j in range(space_num - 1):  
                    file.write(" " + ",")


    def reveal_all(self, *args):       
    	# reveal some results based on args
        if len(args) > 0:
            for key in args:
                assert key in self.result_dict.keys()
                print(str(key) , ": " , self.result_dict[key])
        # reveal all results
        else:
            for key in self.result_dict.keys():
                print(str(key) , ": " , self.result_dict[key])

    def write_final(self, save_path=os.getcwd(), save_name='result_all.csv'):
        fl=open(save_path + '/' + save_name, 'w')
        for key in self.result_dict.keys():
            space_num = self.result_dict[key][0].size
            value_num = len(self.result_dict[key])
            # print(space_num)
            fl.write(str(key) + ",")
            for j in range(space_num - 1):      
                fl.write(" " + ",")
        fl.write("\n")
        # write value
        for iter_key in range(value_num):
            for key in self.result_dict.keys():
                space_num = self.result_dict[key][0].size  
                if space_num == 1:
                    fl.write(str(self.result_dict[key][iter_key]) + ",")
                else:
                    for j in range(space_num):
                        fl.write(str(self.result_dict[key][iter_key][j]) + ",") # TODO-error
                        # print(str(self.result_dict[key][iter_key][j]))
            fl.write("\n")
        fl.close()


# # 使用举例：
# Rcd = Logger()

# for i in range(5):
#     a = np.random.randn(1)
#     b = np.random.randn(2)
#     c = np.random.randn(1)
#     Rcd.store_result(resA=i)
#     Rcd.store_result(resB=b)
#     Rcd.store_result(resC=c)
#     if (i+1) % 2 == 0:
#         # Rcd.write_last(write_period=2)
#         print(a)
#         print("last")
#         Rcd.reveal_last("resA")
# Rcd.write_final()

# a = np.array(np.random.randn(1,2))
# print(type(a))
# print(a)
# print(a.shape)
# a = []
# for j in range(10):
#     a.append(j)
#     if (j+1) % 2 == 0:
#         print("a",a)
#         print("seg:",a[-2:])

    
# print(a)
# # print(a[1:3])
# # print(np.mean(a[1:3]))
# k = 3
# print(a[-k:])