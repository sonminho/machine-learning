from bridge import Bridge
from knight import Knight

print("시뮬레이션을 시작한다.")

bridge = Bridge()
Knight(bridge, "홍길동", "홍천").start()
Knight(bridge, "임꺽정", "임실").start()
Knight(bridge, "일지매", "일산").start()
Knight(bridge, "장보고", "장흥").start()
Knight(bridge, "이순신", "이천").start()