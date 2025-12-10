#include "Fusion/device/Device.h"
#include "Fusion/device/DeviceType.h"

int main() {
   Device device(DeviceType::CPU, 0);
   return 0;
}
