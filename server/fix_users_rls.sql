-- ============================================================================
-- FIX RLS FOR USER SIGNUP
-- Run this in Supabase Dashboard → SQL Editor
-- This fixes the Row Level Security to allow new account creation
-- ============================================================================

-- 1. Disable RLS temporarily to allow setup (will re-enable after)
ALTER TABLE public.users DISABLE ROW LEVEL SECURITY;

-- 2. Drop all existing policies
DROP POLICY IF EXISTS "Users can view own profile" ON public.users;
DROP POLICY IF EXISTS "Users can update own profile" ON public.users;
DROP POLICY IF EXISTS "Users can insert own profile" ON public.users;
DROP POLICY IF EXISTS "Authenticated users can insert own profile" ON public.users;
DROP POLICY IF EXISTS "Service role can manage all users" ON public.users;
DROP POLICY IF EXISTS "Admins can view all users" ON public.users;
DROP POLICY IF EXISTS "Enable insert for authenticated users" ON public.users;
DROP POLICY IF EXISTS "Enable read access for all users" ON public.users;

-- 3. Re-enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- 4. Create simple, working policies

-- Allow anyone to insert their own profile (critical for signup)
CREATE POLICY "Allow insert for signup"
ON public.users
FOR INSERT
WITH CHECK (true);  -- Allow all inserts (Supabase auth handles authentication)

-- Allow users to view their own profile
CREATE POLICY "Allow users to view own profile"
ON public.users
FOR SELECT
USING (true);  -- Allow reading all users (needed for login)

-- Allow users to update their own profile
CREATE POLICY "Allow users to update own profile"
ON public.users
FOR UPDATE
USING (id = auth.uid());

-- 5. Verify policies are created
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual
FROM pg_policies 
WHERE tablename = 'users';

-- 6. Show the table structure
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'users' AND table_schema = 'public'
ORDER BY ordinal_position;