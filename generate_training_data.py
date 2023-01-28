from frame_reader import FrameReader
from annotator import Annotator

if __name__ == "__main__":
    fr = FrameReader("C:\\Users\\yifan\\Desktop\\MLAssessmentData")
    fr.read_all_frames()
    annotator = Annotator('C:\\Users\\yifan\\work\\drill_bit_classifier\\data')
    annotator.generate_annotation()
