LOCAL_PATH := $(call my-dir)

# 第三方预编译的库
include $(CLEAR_VARS)
LOCAL_MODULE := opencv
LOCAL_SRC_FILES := arm64-v8a/libopencv_world.so
LOCAL_EXPORT_C_INCLUDES := opencv2
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS)
LOCAL_MODULE := main
LOCAL_SRC_FILES := main.cpp Tracker.cpp ORBextractor.cc AffineParamEstimator.cpp VisualUtils.cpp
LOCAL_C_INCLUDES := opencv2
LOCAL_SHARED_LIBRARIES := opencv
LOCAL_LDLIBS += -llog -std=c++11
include $(BUILD_EXECUTABLE)