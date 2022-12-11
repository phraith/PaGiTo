class LineProfile {
    start:  IntCoordinate;
    end: IntCoordinate;
    constructor(start: IntCoordinate, end: IntCoordinate) {
        this.start = start;
        this.end = end;
    }

    inverseHeight(height: number): LineProfile
    {
        let startWithInverseHeight = new IntCoordinate(this.start.x, height - this.start.y);
        let endWithInverseHeight = new IntCoordinate(this.end.x, height - this.end.y);
        return new LineProfile(startWithInverseHeight, endWithInverseHeight)
    }
}

enum LineMode {
    Vertical,
    Horizontal
  }

class IntCoordinate {
    x: number;
    y: number;

    constructor(x: number, y: number) {
        this.x = Math.ceil(x);
        this.y = Math.ceil(y);
    }
}

class LineProfileState {
    lineMode: LineMode;
    lineProfiles: LineProfile[];
    currentLineProfile: LineProfile;
    constructor(lineMode: LineMode, lineProfiles: LineProfile[], currentLineProfile: LineProfile)
    {
        this.lineMode = lineMode;
        this.lineProfiles = lineProfiles;
        this.currentLineProfile = currentLineProfile;
    }
}


export { LineProfile, IntCoordinate as Coordinate, LineProfileState, LineMode }