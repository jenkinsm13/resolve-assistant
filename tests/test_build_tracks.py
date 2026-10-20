from resolve_assistant.resolve_build import _ensure_video_tracks, _verify_video_tracks


class FakeTimeline:
    def __init__(self, tracks=1, placed=None):
        self.tracks, self.placed = tracks, placed or {}

    def GetTrackCount(self, kind):
        return self.tracks

    def AddTrack(self, kind):
        self.tracks += 1

    def GetItemListInTrack(self, kind, idx):
        return [object()] * self.placed.get(idx, 0)


ITEMS = [{"trackIndex": 1}] * 3 + [{"trackIndex": 2}] * 6


def test_creates_missing_video_tracks():
    tl = FakeTimeline(tracks=1)
    _ensure_video_tracks(tl, ITEMS)
    assert tl.tracks == 2


def test_no_extra_track_for_single_track_plan():
    tl = FakeTimeline(tracks=1)
    _ensure_video_tracks(tl, [{"trackIndex": 1}])
    assert tl.tracks == 1


def test_reports_dropped_track_two_cuts():
    # the Kashmir failure: V1 got its 3 clips, all 6 V2 cutaways vanished
    assert _verify_video_tracks(FakeTimeline(2, {1: 3, 2: 0}), ITEMS) == ["V2 has 0/6"]


def test_clean_build_has_no_problems():
    assert _verify_video_tracks(FakeTimeline(2, {1: 3, 2: 6}), ITEMS) == []
