#include "network.h"

int main()
{
    CtrlParam param;
    Recver rec;
    if (!rec.setup())
    {
        cout << "FAIL to setup socket!" << endl;
        return 0;
    }
    if (!rec.getvec(param))
        cout << "FAIL to receieve params!" << endl;
    else
        cout << "x= " << param.x << " y= " << param.y << endl;

    if (!rec.getvec(param))
        cout << "FAIL to receieve params!" << endl;
    else
        cout << "x= " << param.x << " y= " << param.y << endl;

    if (!rec.getvec(param))
        cout << "FAIL to receieve params!" << endl;
    else
        cout << "x= " << param.x << " y= " << param.y << endl;
        
    return 0;
}