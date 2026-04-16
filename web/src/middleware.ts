import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const OPEN_PATHS = [
  "/login",
  "/_next",
  "/favicon.ico",
  "/public",
  "/backend-api/auth",
  "/backend-api/health",
  "/backend-api/ready",
];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (OPEN_PATHS.some((p) => pathname === p || pathname.startsWith(`${p}/`))) {
    return NextResponse.next();
  }
  const sessionCookie = request.cookies.get("gary_session");
  if (!sessionCookie?.value) {
    const loginUrl = new URL("/login", request.url);
    return NextResponse.redirect(loginUrl);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|.*\\..*).*)"],
};
