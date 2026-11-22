# 영상과 대형 CSV의 로컬 경로

대회 전술 원본은 `data/tactical/BAR-ATH_tactical.mp4`다. 기존 `mp4/BAR-ATH_tactical-001.mp4`는 로컬 원본 참조다. 중계 자료와 전술 자료에는 각각의 시간축을 사용한다. 외부 경기 자료 루트는 `MTAH_DATA_DIR`로 지정할 수 있다.

SNP 리뷰의 렌더 입력, 영상과 프레임 매핑은 로컬 `공모전archive`에 둔다. JSH의 `data/snp_video_review` 호환 경로는 아카이브의 리뷰 폴더를 가리킨다. 아카이브의 `SNP.csv`는 리뷰 입력이고, PathCRF 추론 재현의 `SNP.input.csv`는 아래 모델 문서에서 확인한다. [모델 재현](reproducibility.md)

대형 프레임별 CSV는 로컬에 둔다. 압축하는 경우 해제한 파일의 SHA-256과 프레임 번호를 대조한다. GitHub에서 가상본이 필요한 테스트는 독립적으로 만든 합성 피치·도형을 사용한다. 원본의 자른 영상이나 좌표를 공유하는 기준은 주최측 규정을 따른다. [규정](../competition/RULES.md)
